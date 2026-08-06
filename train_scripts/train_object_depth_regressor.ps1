$ErrorActionPreference = "Stop"

$DETAIL_ROOT = "D:/lotus/data/hypersim_yolo_detections/train"
$MANIFEST = "D:/lotus/data/hypersim_yolo_detections/train_manifest_score0.5.json"
$OUTPUT = "output/object_depth_regressor"

Write-Host "Training object depth regressor (Hypersim train ONLY)"

python train_object_depth_regressor.py `
  --detail_root=$DETAIL_ROOT `
  --manifest=$MANIFEST `
  --output_dir=$OUTPUT
