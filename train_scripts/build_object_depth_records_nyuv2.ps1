$ErrorActionPreference = "Stop"

$RGB_DIR = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
$DETECTIONS_ROOT = "D:/lotus/data/nyuv2_detail_artifacts/test"
$OUTPUT_ROOT = "D:/lotus/data/object_depth_records_v2/nyuv2_test"
$SCORE_THR = 0.5

Write-Host "Building NYUv2 object depth records (eval only — not for training)"

python utils/build_object_depth_records.py `
  --rgb_dir=$RGB_DIR `
  --detections_root=$DETECTIONS_ROOT `
  --output_root=$OUTPUT_ROOT `
  --pattern="rgb_*.png" `
  --detection_score_thr=$SCORE_THR `
  --dataset=nyuv2_test `
  --depth_unit=mm `
  --skip_existing
