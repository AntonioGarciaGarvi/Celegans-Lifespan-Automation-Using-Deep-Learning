import numpy as np
import cv2
from xml.dom.minidom import parse
import re
from math import pow, sqrt, floor
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import List, Tuple, Dict, Optional
from torch import nn, Tensor
from torchvision.transforms import functional as F
from torchvision.transforms import transforms as T

class ToTensor(nn.Module):
    ##### Class downloaded from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html##
    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        image = F.convert_image_dtype(image)
        return image, target


## Functions to sort folder, files in the "natural" way:
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def extract_coordinates_xml_by_tag(path, tagname):
    # parse an xml file by name
    docA = parse(path)  # xml
    itemsA = docA.getElementsByTagName(tagname)
    LintA = []
    if itemsA[0].firstChild != None:
        valorsA = itemsA[0].firstChild.nodeValue
        valorsA = str(" ".join(valorsA.split()))
        LintA = list(map(int, valorsA.split()))
    return LintA

## Calculate centroids given the bounding boxes
def get_worms_centroid_NN_prediction(NN_pred, score_threshold):
    all_list = []
    for j in range(len(NN_pred[0]['boxes'])):
        if NN_pred[0]['scores'][j] > score_threshold:
            x1, x2, x3, x4 = map(int, NN_pred[0]['boxes'][j].tolist())
            xc = x1 + int((x3-x1)/2)
            yc = x2 + int((x4-x2)/2)
            all_list.append(xc)
            all_list.append(yc)

    nf = int(len(all_list) / 2)
    all_matrix = np.array(all_list).reshape(nf, 2)
    return all_matrix

## Function to generate synthetic images from real image celegans location
def generate_circles_img(worms_coord, real_size=1944):
    ## background parameters
    height = 256
    width = 256
    background_color = 60

    ## Objects (circles) parameters
    radius = 3
    color = 0  # (255, 0, 0)
    thickness = -1  # thickness of -1 px will fill the circle shape by the specified color.
    # generate the background image
    background = background_color * np.ones((height, width), np.uint8)

    scaley = real_size / height
    scalex = real_size / width

    for worm in worms_coord:
        cx, cy = worm
        cx /= scalex
        cy /= scaley

        background = cv2.circle(background, (int(cx), int(cy)), radius, color, thickness)

    return background

# pre-trained model
# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
def get_model_instance(num_classes):
    # load an instance model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# -------------------------Functions to read CMPR files----------------------------------#
def my_memcpy(dest, src, size, pd, ps):
    for i in range(0, size, 1):
        dest[pd] = src[ps]
        ps += 1
        pd += 1
    return dest


def my_memset(des, val, nb, p):
    while nb > 0:
        des[p] = val
        p += 1
        nb -= 1
    return des


def reconstruir_circul(digital1, ample, alt):
    r = alt / 2
    y0 = alt / 2
    k2 = 0
    x0 = ample / 2
    r2 = pow(r, 2)

    NUM = y0 - r
    if NUM < 0:
        NUM = 0
    i1 = int(NUM * ample)
    digital2 = [0] * int(pow(1944, 2))

    for j in range(0, alt, 1):
        xs = int(floor(x0 - abs(sqrt(r2 - pow(r - j, 2)))))
        cx = int(floor(abs(sqrt(r2 - pow(r - j, 2)))))
        NUM = j * ample
        N1 = NUM + ample
        XN = NUM + xs
        Xc = XN + 2 * cx
        digital2 = my_memset(digital2, 100, XN - i1, i1)
        i1 = XN - 1
        digital2 = my_memcpy(digital2, digital1, Xc - XN, i1, k2)
        i1 = i1 + Xc - XN
        k2 = k2 + Xc - XN
        digital2 = my_memset(digital2, 100, N1 - i1, i1)
        i1 = N1 - 1

    return np.asarray(digital2)


def create_image_from_cmpr(image_path):
    with open(image_path, mode='rb') as file:
        fileContent = file.read()

    np_im = reconstruir_circul(fileContent, 1944, 1944)
    np_im = np_im.reshape(1944, 1944)
    new_im = Image.fromarray(np_im.astype('uint8'), 'L')
    opencv_im = np.array(new_im)
    return opencv_im

# -----------------------------------------------------------#
