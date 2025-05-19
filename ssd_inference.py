import os
import glob
from pathlib import Path
import json
import subprocess
import time

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

import hydra

import logging
log = logging.getLogger(__name__)

def make_yolo_annotation(img_shape, bbox_pts, score):
    xmin = bbox_pts[0]
    ymin = bbox_pts[1]
    xmax = bbox_pts[2]
    ymax = bbox_pts[3]
    h, w = img_shape[0], img_shape[1]
    x_c = ((xmin + xmax) / 2) / w
    y_c = ((ymin + ymax) / 2) / h
    width = (xmax - xmin) / w
    height = (ymax - ymin) / h
    return f"{0} {x_c:.6f} {y_c:.6f} {width:.6f} {height:.6f} {score:.6f}"

@hydra.main(version_base=None, config_path='config', config_name='ssd_detect')
def run(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_path = hydra_cfg['runtime']['output_dir']
    save_img_to = os.path.join(run_path, 'images')
    save_lbl_to = os.path.join(run_path, 'labels')
    os.makedirs(save_img_to, exist_ok=True)
    os.makedirs(save_lbl_to, exist_ok=True)
    imgs = glob.glob(os.path.join(cfg['imgs'], '**.jpg'))
    log.info(f'FOUND {len(imgs)} images')

    net = build_ssd('test', 300, 2)    # initialize SSD
    net.load_weights(cfg['weights'])

    preproc_time_inf = []
    model_time_inf = []
    postproc_time_inf = []
    for img in imgs:
        img_name = Path(img).stem

        img = cv2.imread(img)
        st = time.time()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        x = cv2.resize(img, (300, 300)).astype(np.float32)
        x -= (104.0, 117.0, 123.0)
        x = x.astype(np.float32)
        x = x[:, :, ::-1].copy()
        x = torch.from_numpy(x).permute(2, 0, 1)
        preproc_time_inf.append(time.time() - st)

        xx = x.unsqueeze(0)
        if torch.cuda.is_available():
            xx = xx.cuda()
        st = time.time()
        y = net(xx)
        model_time_inf.append(time.time() - st)

        st = time.time()
        detections = y.data
        scale = torch.Tensor(img.shape[1::-1]).repeat(2)

        plt.figure(figsize=(10,10))
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.imshow(img)  # plot the image for matplotlib
        currentAxis = plt.gca()

        annots = []
        if cfg['apply_nms'] == False:
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i,j, 0] > cfg['conf_thres']:
                    score = detections[0,i,j,0]
                    pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))
                    display_txt = '%s: %.2f'%('logo', score)
                    currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':'red', 'alpha':0.5})
                    j += 1
                    annots.append(make_yolo_annotation(img.shape[:-1], pt, score))
        else:
            try:
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
                # log.info(scores)
                for i in range(scores.size(0)):
                    score = scores[i].item()
                    pt = boxes[i].cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                    currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))
                    display_txt = '%s: %.2f'%('logo', score)
                    currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':'red', 'alpha':0.5})
                    annots.append(make_yolo_annotation(img.shape[:-1], pt, score))
            except Exception as e:
                pass


        postproc_time_inf.append(time.time() - st)
        
        with open(os.path.join(save_lbl_to, f'{img_name}.txt'), "w") as f:
            f.write("\n".join(annots))
        plt.savefig(os.path.join(save_img_to, f'{img_name}.png'))

    print(f'PREPROC: {np.array(preproc_time_inf).mean()}')
    print(f'MODEL: {np.array(model_time_inf).mean()}')
    print(f'POSTPROC: {np.array(postproc_time_inf).mean()}')

if __name__ == '__main__':
    run()