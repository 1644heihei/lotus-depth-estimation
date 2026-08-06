$ErrorActionPreference = "Stop"

$ARCHIVE = "datasets/eval/depth/scannet/scannet_val_sampled_800_1.tar"
$SPLIT = "datasets/eval/depth/data_split/scannet/scannet_val_sampled_list_800_1.txt"
$RGB_ROOT = "D:/lotus/data/scannet_depth_eval_v1"
$YOLO_ROOT = "D:/lotus/data/scannet_yolo_artifacts_v1"
$RECORD_ROOT = "D:/lotus/data/object_depth_records_v2/scannet_eval"

if (-not (Test-Path $ARCHIVE)) {
  throw "Missing official ScanNet archive: $ARCHIVE"
}

python utils/prepare_scannet_depth_eval.py `
  --archive=$ARCHIVE `
  --split_list=$SPLIT `
  --output_dir=$RGB_ROOT

python utils/build_detail_train_dataset.py `
  --rgb_dir=$RGB_ROOT `
  --output_dir=$YOLO_ROOT `
  --core_model=jingheya/lotus-depth-d-v2-0-disparity `
  --steps=yolo `
  --pattern="*.jpg" `
  --yolo_score_thr=0.5 `
  --yolo_device=cpu

python utils/build_object_depth_records.py `
  --rgb_dir=$RGB_ROOT `
  --detections_root=$YOLO_ROOT `
  --output_root=$RECORD_ROOT `
  --pattern="*.jpg" `
  --detection_score_thr=0.5 `
  --dataset=scannet_eval `
  --depth_unit=mm `
  --skip_existing
