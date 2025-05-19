import os
import glob
from pathlib import Path
import json
import subprocess
import time
import argparse
from tqdm import tqdm

import sys
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import cv2
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.set_default_device(1)

from ssd import build_ssd

import hydra

import logging
log = logging.getLogger(__name__)

def run(cfg):
    net = build_ssd('test', 300, 2)
    net.load_weights(cfg.weights)
    net = net.cuda()
    net.eval()

    img = np.random.randn(300, 300, 3).astype(np.float32)
    img -= (104.0, 117.0, 123.0)
    img = torch.from_numpy(img).permute(2, 0, 1)

    img = torch.stack([img for _ in range(int(cfg.batch_size))], axis=0).cuda()

    times = []
    with torch.no_grad():
        for i in tqdm(range(100)):
            st_time = time.time()
            _ = net(img)
            end_time = time.time()
            times.append(end_time - st_time)
    print(f'AVG INFERENCE TIME: {np.array(times)[10:].mean()} sec')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str)
    parser.add_argument('--batch_size', type=str)

    opt_parser = parser.parse_args()
    run(opt_parser)