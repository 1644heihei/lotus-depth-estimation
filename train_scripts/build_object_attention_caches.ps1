$ErrorActionPreference = "Stop"
Set-Location "D:/lotus/lotus-depth-estimation"

$REGRESSOR = "output/object_depth_regressor_v5_roi"
$OUT_DIR = "D:/lotus/data/object_attention_cache_v2_roi"

python utils/build_object_attention_cache.py `
  --rgb_dir="D:/lotus/data/hypersim_processed/train" `
  --detections_root="D:/lotus/data/hypersim_yolo_detections/train" `
  --regressor_dir=$REGRESSOR `
  --manifest="D:/lotus/data/hypersim_yolo_detections/train_manifest_score0.5.json" `
  --output="$OUT_DIR/hypersim_train.npz" `
  --detection_score_thr=0.5
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python utils/build_object_attention_cache.py `
  --rgb_dir="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test" `
  --detections_root="D:/lotus/data/nyuv2_detail_artifacts/test" `
  --regressor_dir=$REGRESSOR `
  --pattern="rgb_*.png" `
  --output="$OUT_DIR/nyuv2_test.npz" `
  --detection_score_thr=0.5
exit $LASTEXITCODE
