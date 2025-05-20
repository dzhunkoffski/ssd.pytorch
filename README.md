# Prepare environment
```bash
conda env create -f environment.yaml
```
# Run training examle:
example:
```bash
CUDA_VISIBLE_DEVICES=2 python train.py --dataset COCO --dataset_root data/processed/LogoDet1CocoV4_filter --batch_size 16 --num_workers 2 --lr 0.0003 --save_folder datasetv4_filter_miniou07
```
where LogoDet1CocoV4_filter is a dataset that follows COCO format

# Inference
example:
```bash
python ssd_inference.py run_name=RUN_NAME weights=PATH_TO_PTH_CHECKPOINT 
```

# Fix NMS bounding box (along rows or columns)
example:
```bash
python ssd_fix_bboxes.py run_name=RUN_NAME img_path=gambling-ad-detection.v2i.yolov9/train/images labels=outputs/2025-04-28/23-15-25___default_long/labels how=row
```
* `img_path` - path to clean images without bounding boxes, so images with new bboxes will be created
* `how` can be either row or column
