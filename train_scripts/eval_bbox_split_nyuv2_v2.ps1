$ErrorActionPreference = "Stop"

# In/out-of-bbox abs_rel split across the fine-grained v2 retrain
# (train_lotus_d_object_bbox_loss_v2.ps1), which keeps every 250-step
# checkpoint from 250..3000.
$DETAIL_MODEL = "output/train-lotus-d-object-bbox-lora-v2"
$REGRESSOR = "output/object_depth_regressor_v5_roi"
$ARTIFACTS = "D:/lotus/data/nyuv2_detail_artifacts/test"
$OUT_ROOT = "output/eval_bbox_split_object_bbox_lora_v2"

$STEPS = 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000

foreach ($step in $STEPS) {
  $summaryPath = "$OUT_ROOT/checkpoint-$step/summary.json"
  if (Test-Path $summaryPath) {
    Write-Host "=== bbox-split eval: checkpoint-$step (skipped, summary.json exists) ==="
    continue
  }
  Write-Host "=== bbox-split eval: checkpoint-$step ==="
  python eval_regressor_predepth_nyuv2.py `
    --detail_model=$DETAIL_MODEL `
    --checkpoint_step=$step `
    --regressor_dir=$REGRESSOR `
    --detail_artifacts_dir=$ARTIFACTS `
    --condition_mode=unconditioned `
    --detection_score_thr=0.5 `
    --processing_res=512 `
    --seed=42 `
    --half_precision `
    --output_dir="$OUT_ROOT/checkpoint-$step"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python summarize_bbox_split_eval.py --sweep_dir=$OUT_ROOT --csv_out="$OUT_ROOT/summary.csv"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
exit 0
