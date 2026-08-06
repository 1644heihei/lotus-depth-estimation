$ErrorActionPreference = "Stop"

$RGB_DIR = "D:/lotus/data/hypersim_processed/train"
$DETECTIONS_ROOT = "D:/lotus/data/hypersim_yolo_detections/train"
$OUTPUT_ROOT = "D:/lotus/data/object_depth_records_v2/hypersim_train"
$MANIFEST = "D:/lotus/data/hypersim_yolo_detections/train_manifest_score0.5.json"
$SCORE_THR = 0.5

Write-Host "Building Hypersim object depth records (train only)"

python utils/build_object_depth_records.py `
  --rgb_dir=$RGB_DIR `
  --detections_root=$DETECTIONS_ROOT `
  --output_root=$OUTPUT_ROOT `
  --manifest=$MANIFEST `
  --detection_score_thr=$SCORE_THR `
  --dataset=hypersim_train `
  --depth_unit=mm `
  --skip_existing
