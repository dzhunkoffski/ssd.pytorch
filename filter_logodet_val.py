import os
import glob
from pathlib import Path
import json
import subprocess
import time
from tqdm import tqdm
from PIL import Image

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

from ssd import build_ssd

DATASET_PATH = '/home/stud_lab_vk_01/ad-detection/data/processed/LogoDet1YoloV4_filter'
os.makedirs(os.path.join(DATASET_PATH, 'removed_imgs'), exist_ok=True)
val_imgs = glob.glob(os.path.join(DATASET_PATH, 'images', 'val', '**.jpg'))
before_filter = len(val_imgs)

WEIGHT_PATH = '/home/stud_lab_vk_01/ssd.pytorch/default_weights/ssd300_COCO_395000.pth'
net = build_ssd('test', 300, 2)
net.load_weights(WEIGHT_PATH)

# clean train
PART = 'val'
THRESHOLD = 0.3
n_deleted = 0
for img_p in tqdm(val_imgs):
    img_name = Path(img_p).stem
    label_path = os.path.join(DATASET_PATH, 'labels', PART, f'{img_name}.txt')
    try:
        with open(label_path, 'r') as fd:
            boxes = fd.readlines()
        n_boxes = len(boxes)

        # read img and get number of predicted boxes
        img = cv2.imread(img_p)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(img, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = x.unsqueeze(0)
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = net(xx)
        detections = y.data
        scale = torch.Tensor(img.shape[1::-1]).repeat(2)
        scores = []
        boxes = []
        for i in range(detections.size(1)):
            for j in range(detections.size(2)):
                score = detections[0,i,j,0]
                if score > 0.2:
                    pt = (detections[0,i,j,1:]*scale).cpu().squeeze().unsqueeze(0)
                    boxes.append(pt)
                    scores.append(score)
        scores = torch.tensor(scores)
        boxes = torch.cat(boxes, dim=0)
        boxes_ixs = torchvision.ops.nms(boxes.cpu(), scores.cpu(), iou_threshold=0.2)
        
        boxes = boxes[boxes_ixs]
        scores = scores[boxes_ixs]
        good_scores = scores[scores > THRESHOLD]
        if len(good_scores) - n_boxes > 1:
            n_deleted += 1
            img = Image.open(img_p)
            img.save(os.path.join(DATASET_PATH, 'removed_imgs', f'{img_name}.png'))
            os.remove(img_p)
            os.remove(label_path)
    except Exception as e:
        print('Remove', img_name)
        if os.path.exists(img_p):
            os.remove(img_p)
        if os.path.exists(label_path):
            os.remove(label_path)

after_filter = len(glob.glob(os.path.join(DATASET_PATH, 'images', 'val', '**.jpg')))
print(f'BEFORE: {before_filter}; AFTER: {after_filter}')
