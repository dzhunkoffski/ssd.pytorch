import os
import glob
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import cv2

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

def mid2corner(bbox):
    return (
        bbox[0] - 0.5 * bbox[2],
        bbox[1] - 0.5 * bbox[3],
        bbox[0] + 0.5 * bbox[2],
        bbox[1] + 0.5 * bbox[3]
    )

def row_merge_labels_bruteforce(yolo_labels: List[str], row_threshold: float, union_threshold: float):
    corner_bboxes = []
    for line in yolo_labels:
        corner_bboxes.append(mid2corner(list(map(float, line.split()[1:]))))

    current_bboxes = corner_bboxes.copy()
    prev_len = len(current_bboxes) + 1
    while len(current_bboxes) != prev_len:
        prev_len = len(current_bboxes)
        index_to_remove = []
        pairs_to_merge = []
        for i in range(len(current_bboxes)):
            for j in range(i+1, len(current_bboxes)):
                flag1 = i != j
                flag2 = abs(current_bboxes[i][1] - current_bboxes[j][1]) < row_threshold and abs(current_bboxes[i][3] - current_bboxes[j][3]) < row_threshold
                flag3 = abs(current_bboxes[i][0] - current_bboxes[j][2]) < union_threshold or abs(current_bboxes[i][2] - current_bboxes[j][0]) < union_threshold

                if flag1 and flag2 and flag3:
                    index_to_remove.append(i)
                    index_to_remove.append(j)
                    pairs_to_merge.append((i, j))
        index_to_remove = set(index_to_remove)
        new_bboxes = []
        for i,j in pairs_to_merge:
            new_bboxes.append([
                min(current_bboxes[i][0], current_bboxes[j][0]),
                min(current_bboxes[i][1], current_bboxes[j][1]),
                max(current_bboxes[i][2], current_bboxes[j][2]),
                max(current_bboxes[i][3], current_bboxes[j][3])
            ])

        t = [current_bboxes[i] for i in range(len(current_bboxes)) if i not in index_to_remove]
        current_bboxes = t
        current_bboxes += new_bboxes
    return current_bboxes + corner_bboxes

def column_merge_labels_bruteforce(yolo_labels: List[str], row_threshold: float, union_threshold: float):
    corner_bboxes = []
    for line in yolo_labels:
        corner_bboxes.append(mid2corner(list(map(float, line.split()[1:]))))

    current_bboxes = corner_bboxes.copy()
    prev_len = len(current_bboxes) + 1
    while len(current_bboxes) != prev_len:
        prev_len = len(current_bboxes)
        index_to_remove = []
        pairs_to_merge = []
        for i in range(len(current_bboxes)):
            for j in range(i+1, len(current_bboxes)):
                flag1 = i != j
                flag2 = abs(current_bboxes[i][0] - current_bboxes[j][0]) < row_threshold and abs(current_bboxes[i][2] - current_bboxes[j][2]) < row_threshold
                flag3 = abs(current_bboxes[i][1] - current_bboxes[j][3]) < union_threshold or abs(current_bboxes[i][3] - current_bboxes[j][1]) < union_threshold

                if flag1 and flag2 and flag3:
                    index_to_remove.append(i)
                    index_to_remove.append(j)
                    pairs_to_merge.append((i, j))
        index_to_remove = set(index_to_remove)
        new_bboxes = []
        for i,j in pairs_to_merge:
            new_bboxes.append([
                min(current_bboxes[i][0], current_bboxes[j][0]),
                min(current_bboxes[i][1], current_bboxes[j][1]),
                max(current_bboxes[i][2], current_bboxes[j][2]),
                max(current_bboxes[i][3], current_bboxes[j][3])
            ])

        t = [current_bboxes[i] for i in range(len(current_bboxes)) if i not in index_to_remove]
        current_bboxes = t
        current_bboxes += new_bboxes
    return current_bboxes + corner_bboxes

def row_merge_labels(yolo_labels: List[str], row_threshold: float, union_threshold: float):
    corner_bboxes = []
    for line in yolo_labels:
        corner_bboxes.append(mid2corner(list(map(float, line.split()[1:]))))
    
    # Group boxes by row
    rows = []
    for box in corner_bboxes:
        placed = False
        for i in range(len(rows)):
            # Check vertical overlap or closeness
            if abs(box[1] - rows[i][0][1]) < row_threshold or abs(box[3] - rows[i][0][3]) < row_threshold:
                rows[i].append(box)
                placed = True
                break
        if not placed:
            rows.append([box])

    merged_boxes = []
    # Merge bboxes in every row i they are close
    for row in rows:
        row = sorted(row, key=lambda b: b[0])  # sort by x_min
        merged = []
        current = row[0]
        for next_box in row[1:]:
            if next_box[0] - current[2] < union_threshold:
                # Merge
                current = [
                    min(current[0], next_box[0]),
                    min(current[1], next_box[1]),
                    max(current[2], next_box[2]),
                    max(current[3], next_box[3])
                ]
            else:
                merged.append(current)
                current = next_box
        merged.append(current)
        merged_boxes.extend(merged)
    return merged_boxes

def column_merge_labels(yolo_labels: List[str], row_threshold: float, union_threshold: float):
    corner_bboxes = []
    for line in yolo_labels:
        corner_bboxes.append(mid2corner(list(map(float, line.split()[1:]))))
    
    # Group boxes by row
    columns = []
    for box in corner_bboxes:
        placed = False
        for i in range(len(columns)):
            # Check vertical overlap or closeness
            if abs(box[0] - columns[i][0][0]) < row_threshold or abs(box[2] - columns[i][0][2]) < row_threshold:
                columns[i].append(box)
                placed = True
                break
        if not placed:
            columns.append([box])

    merged_boxes = []
    # Merge bboxes in every row i they are close
    for col in columns:
        col = sorted(col, key=lambda b: b[0])  # sort by x_min
        merged = []
        current = col[0]
        for next_box in col[1:]:
            if next_box[3] - current[1] < union_threshold:
                # Merge
                current = [
                    min(current[0], next_box[0]),
                    min(current[1], next_box[1]),
                    max(current[2], next_box[2]),
                    max(current[3], next_box[3])
                ]
            else:
                merged.append(current)
                current = next_box
        merged.append(current)
        merged_boxes.extend(merged)
    return merged_boxes

@hydra.main(version_base=None, config_path='config', config_name='ssd_fix_bbox')
def run(cfg):
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    run_path = hydra_cfg['runtime']['output_dir']
    save_images_to = os.path.join(run_path, 'images')
    save_labels_to = os.path.join(run_path, 'labels')
    os.makedirs(save_images_to, exist_ok=True)
    os.makedirs(save_labels_to, exist_ok=True)
    labels = glob.glob(os.path.join(cfg['labels'], '**.txt'))
    log.info(f'FOUND {len(labels)} labels')

    for lbl_file in tqdm(labels):
        fname = Path(lbl_file).stem
        img_path = os.path.join(cfg['img_path'], f'{fname}.jpg')
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        plt.figure(figsize=(10,10))
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.imshow(img)  # plot the image for matplotlib
        currentAxis = plt.gca()
        with open(lbl_file, 'r') as fd:
            lbl_lines = fd.readlines()
            if cfg['how'] == 'row':
                new_bboxes = row_merge_labels_bruteforce(lbl_lines, cfg['split_threshold'], cfg['union_threshold'])
            elif cfg['how'] == 'column':
                new_bboxes = column_merge_labels(lbl_lines, cfg['split_threshold'], cfg['union_threshold'])
        annots = []
        for bbox in new_bboxes:
            bbox = (bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height)
            coords = (bbox[0], bbox[1]), (bbox[2]-bbox[0]+1), (bbox[3]-bbox[1]+1)
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))
            display_txt = 'logo'
            currentAxis.text(bbox[0], bbox[1], display_txt, bbox={'facecolor':'red', 'alpha':0.5})
            annots.append(make_yolo_annotation(img.shape[:-1], bbox, 0.0))
        with open(os.path.join(save_labels_to, f'{fname}.txt'), "w") as f:
            f.write("\n".join(annots))
        plt.savefig(os.path.join(save_images_to, f'{fname}.png'))
        plt.close()

if __name__ == '__main__':
    run()
