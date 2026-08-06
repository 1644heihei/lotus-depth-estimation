$ErrorActionPreference = "Stop"

$DETAIL_ROOT = "D:/lotus/data/object_depth_records_v2/hypersim_train"
$MANIFEST = "D:/lotus/data/hypersim_yolo_detections/train_manifest_score0.5.json"
$OUTPUT = "output/object_depth_regressor_v3"

Write-Host "Training object depth regressor v2 (extended features + depth filter)"

python train_object_depth_regressor.py `
  --detail_root=$DETAIL_ROOT `
  --manifest=$MANIFEST `
  --output_dir=$OUTPUT `
  --feature_version=2 `
  --max_depth_m=30 `
  --epochs=80 `
  --patience=12
