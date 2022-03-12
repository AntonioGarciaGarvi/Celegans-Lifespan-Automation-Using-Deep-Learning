import re
import numpy as np
import torch
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt


def atoi(text):
    return int(text) if text.isdigit() else text
def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


def calculate_mean_life(liv_worm_sec):
    dead_worm_sec = []
    days = []
    day = 4
    for index in range(len(liv_worm_sec)):
        days.append(day)
        day += 1
        if index == 0:
            dead_worm_sec.append(0)

        else:
            dead_worm_sec.append(liv_worm_sec[index-1]-liv_worm_sec[index])

    prod = np.array(dead_worm_sec) * np.array(days)
    suma = sum(prod)
    mean_life = suma / liv_worm_sec[0]

    return mean_life


def post_process_filter(real_labels, automatic_labels, filter_limit=False, filter_correction=False,mlife=None):
    processed_labels = torch.zeros(57)
    if mlife is None:
        mlife = int(calculate_mean_life(real_labels))

    if filter_limit: 
        living_worms_first_day = max(real_labels)  
        for lab in range(len(automatic_labels)):
            val = torch.tensor(np.asarray([int(min(automatic_labels[lab], living_worms_first_day))]), dtype=torch.float)
            processed_labels[lab] = val 
    if filter_correction:
        for lab2 in range(len(automatic_labels)):
            if lab2 <= (mlife - 4): 
                x = processed_labels[lab2]
                i=1
                while lab2-i >= 0 and x > processed_labels[lab2-i]:
                    processed_labels[lab2-i] = x
                    i += 1
            else:
                x = processed_labels[lab2]
                if x > processed_labels[lab2-1]:
                    processed_labels[lab2]= processed_labels[lab2-1]
    return processed_labels

def post_process_filter2(real_labels, automatic_labels, filter_limit=False, filter_correction=False,mlife=None):
    processed_labels = torch.zeros(57)
    if mlife is None:
        mlife = int(calculate_mean_life(real_labels))

    if filter_limit: 
        living_worms_first_day = max(real_labels)
        for lab in range(len(automatic_labels)):
            val = torch.tensor(np.asarray([int(min(torch.round(automatic_labels[lab]), living_worms_first_day))]), dtype=torch.float)
            processed_labels[lab] = val 
            
    if filter_correction:
        for lab2 in range(len(automatic_labels)):
            if lab2 <= (mlife - 4): 
                x = processed_labels[lab2]
                i=1
                while lab2-i >= 0 and x > processed_labels[lab2-i]:
                    processed_labels[lab2-i] = x
                    i += 1
            else:
                x = processed_labels[lab2]
                if x > processed_labels[lab2-1]:
                    processed_labels[lab2]= processed_labels[lab2-1]
    return processed_labels


class LifespanDatasetTest(torch.utils.data.Dataset):
    def __init__(self, root_dir, seq_length, transform=None, augmentation=None):

        self.root_dir = root_dir
        self.seq_lenght = seq_length
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return int(len(os.listdir(self.root_dir)))

    def __getitem__(self,idx):
        folders = os.listdir(self.root_dir)
        folders.sort(key=natural_keys)

        subdir = folders[idx]
        labels = torch.zeros(self.seq_lenght)
        list_pics = []
        folders_days = os.listdir(self.root_dir + "/" + subdir)
        folders_days.sort(key=natural_keys)

        days_available = []
        for sub_days_dir in folders_days:
            start = sub_days_dir.find("dia=")
            finish = sub_days_dir.find(" cond")
            days_available.append(int(sub_days_dir[start + len("dia="):finish]))
        if days_available[0] == 1:
            del(folders_days[0])

        day_real = 4
        day = 0
        for file in folders_days:
            start = file.find("dia=")
            finish = file.find(" cond")
            day_available = int(file[start + len("dia="):finish])
            if day_real != day_available:
                day_before_gap = day_real - 1
                day_after_gap = days_available[days_available.index(day_before_gap) + 1]
                gaps = day_after_gap - day_before_gap - 1
                for gap in range(gaps):
                    img = 255 * np.ones((256, 256), np.uint8)
                    img = Image.fromarray(img)
                    img_tens = self.transform(img)
                    list_pics.append(img_tens)
                    labels[day] = no_living_worms
                    day += 1
                    day_real += 1

            img = cv2.imread(self.root_dir + "/" + subdir + "/" + file, 0)
            start = file.find("living_worms=")
            finish = file.find(".jpg")
            label_viv = (file[start + len("living_worms="):finish])

            img = Image.fromarray(img)
            img_tens = self.transform(img)
            list_pics.append(img_tens)
            no_living_worms = torch.tensor(np.asarray([int(label_viv)]), dtype=torch.int)
            labels[day] = no_living_worms
            day += 1
            day_real += 1

        if len(list_pics) < self.seq_lenght:
            for rep in range(0, (self.seq_lenght - len(list_pics))):
                list_pics.append(img_tens)
                no_living_worms = torch.tensor(np.asarray([int(0)]), dtype=torch.int)
                labels[day + rep] = no_living_worms
        labels_pre = labels
        labels = post_process_filter(labels,labels, filter_limit=True, filter_correction=True,mlife=14)
        stacked_set = torch.stack(list_pics)
        composed_sample = [stacked_set, labels, labels_pre, subdir]

        return composed_sample


def plot_real_model_curves(current_labels, pre):
    days = np.arange(4, 61)  
    living_worms_real_per = np.zeros((len(days)))
    living_worms_model_per = np.zeros((len(days)))

    for day in range(len(days)):
        living_worms_at_start = current_labels[0].data.cpu().numpy()
        living_worms_real_per[day] = current_labels[day].data.cpu().numpy() / living_worms_at_start * 100  
        living_worms_model_per[day] = pre[day].data.cpu().numpy() / living_worms_at_start * 100
        
    plt.figure(figsize=(1855 / 96, 986 / 96), dpi=96)
    plt.plot(days, living_worms_real_per, "r-", label="Real data", linewidth=3)
    plt.plot(days, living_worms_model_per, "b-", label="Model results", linewidth=3)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel("Days", fontsize=15, fontweight='bold')
    plt.ylabel("Survival", fontsize=20, fontweight='bold')
    plt.grid(True, axis="y")
    plt.ylim((0, 120))
    plt.legend(fontsize=20)
    plt.show()
