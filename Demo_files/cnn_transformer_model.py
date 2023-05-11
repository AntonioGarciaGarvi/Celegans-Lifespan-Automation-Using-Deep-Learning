import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# Transformer implementation: https://github.com/lucidrains/vit-pytorch
# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class CNN_Transformer(nn.Module):
    def __init__(self, *, seq_length, img_size, kernel_s_layer_1, kernel_s, max1, max2, max3, out_channels_1,
                 out_channels_2, out_channels_final, dim, depth, heads, mlp_dim, dim_head=64, dropout=0.,
                 emb_dropout=0.):
        super().__init__()
        self.seq_length = seq_length
        self.max_1 = max1
        self.max_2 = max2
        self.max_3 = max3

        self.conv1 = nn.Conv2d(1, out_channels_1, kernel_size=kernel_s_layer_1)
        self.cnn_bn1 = nn.BatchNorm2d(out_channels_1)
        self.conv2 = nn.Conv2d(out_channels_1, out_channels_2, kernel_size=kernel_s)
        self.cnn_bn2 = nn.BatchNorm2d(out_channels_2)
        self.conv3 = nn.Conv2d(out_channels_2, out_channels_final, kernel_size=kernel_s)
        self.cnn_bn3 = nn.BatchNorm2d(out_channels_final)

        self.img_size = img_size
        size_after_cnn = int((int((int(
            (img_size - kernel_s_layer_1 + 1) / self.max_1) - kernel_s + 1) / self.max_2) - kernel_s + 1) / self.max_3)
        self.out_size_cnn = out_channels_final * size_after_cnn ** 2
        self.dim = dim

        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.seq_length, self.out_size_cnn))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.fc1 = nn.Linear(in_features=self.seq_length * self.dim, out_features=2000)
        self.bn1 = nn.BatchNorm1d(num_features=2000)
        self.fc2 = nn.Linear(in_features=2000, out_features=2000)
        self.bn2 = nn.BatchNorm1d(num_features=2000)
        self.fc3 = nn.Linear(in_features=2000, out_features=750)
        self.bn3 = nn.BatchNorm1d(num_features=750)
        self.fc4 = nn.Linear(in_features=750, out_features=seq_length)

    def forward(self, t):
        batch_s, timesteps, C, H, W = t.size()
        t = t.view(batch_s * timesteps, 1, H, W)
        t = F.max_pool2d(F.leaky_relu(self.cnn_bn1(self.conv1(t))), kernel_size=self.max_1, stride=self.max_1)
        t = F.max_pool2d(F.leaky_relu(self.cnn_bn2(self.conv2(t))), kernel_size=self.max_2, stride=self.max_2)
        t = F.max_pool2d(F.leaky_relu(self.cnn_bn3(self.conv3(t))), kernel_size=self.max_3, stride=self.max_3)
        t = t.reshape(batch_s, self.seq_length, self.out_size_cnn)

        b, n, _ = t.shape
        t += self.pos_embedding[:, :n]
        t = self.dropout(t)
        t = self.transformer(t)
        t = t.reshape(batch_s, -1)
        t = F.leaky_relu(self.bn1(self.fc1(t)))
        t = F.leaky_relu(self.bn2(self.fc2(t)))
        t = F.leaky_relu(self.bn3(self.fc3(t)))
        t = self.fc4(t)

        return t

