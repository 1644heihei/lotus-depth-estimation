$ErrorActionPreference = "Stop"

# NOTE: processing_res was 512 in the original runs - a misconfiguration inherited from
# the regressor config (see docs/phase0_findings.md 3.3.1). 768 is Lotus's official
# default. Recorded results in docs/ were produced at 512; re-running now gives
# different (better) numbers. Set $PROCESSING_RES=512 to reproduce the historical run.
$PROCESSING_RES = 768

# In/out-of-bbox abs_rel split across the v3 retrain (cosine LR schedule
# experiment, train_lotus_d_object_bbox_loss_v3.ps1). Same checkpoint grid
# as v2 for a direct comparison.
$DETAIL_MODEL = "output/train-lotus-d-object-bbox-lora-v3"
$REGRESSOR = "output/object_depth_regressor_v5_roi"
$ARTIFACTS = "D:/lotus/data/nyuv2_detail_artifacts/test"
$OUT_ROOT = "output/eval_bbox_split_object_bbox_lora_v3"

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
    --processing_res=$PROCESSING_RES `
    --seed=42 `
    --half_precision `
    --output_dir="$OUT_ROOT/checkpoint-$step"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

python summarize_bbox_split_eval.py --sweep_dir=$OUT_ROOT --csv_out="$OUT_ROOT/summary.csv"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
exit 0
