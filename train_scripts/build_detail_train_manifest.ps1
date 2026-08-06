$ErrorActionPreference = "Stop"

# ========= User settings: detection-only training manifest =========
$RGB_DIR = "D:/lotus/data/hypersim_processed/train"
$DETAIL_ROOT = "D:/lotus/data/hypersim_yolo_detections/train"
$OUTPUT = "D:/lotus/data/hypersim_yolo_detections/train_manifest_score0.5.json"
$SCORE_THR = 0.5

# =================================

Write-Host "Building detail-train manifest (images with detections only)"
Write-Host "RGB_DIR: $RGB_DIR"
Write-Host "DETAIL_ROOT: $DETAIL_ROOT"
Write-Host "SCORE_THR: $SCORE_THR"
Write-Host "OUTPUT: $OUTPUT"

python utils/filter_detail_train_manifest.py `
  --rgb_dir=$RGB_DIR `
  --detail_root=$DETAIL_ROOT `
  --output=$OUTPUT `
  --detection_score_thr=$SCORE_THR `
  --require_artifacts

Write-Host "Done. Use with training:"
Write-Host "  --detail_train_manifest=$OUTPUT"
