$ErrorActionPreference = "Stop"

# NOTE: processing_res was 512 in the original runs - a misconfiguration inherited from
# the regressor config (see docs/phase0_findings.md 3.3.1). 768 is Lotus's official
# default. Recorded results in docs/ were produced at 512; re-running now gives
# different (better) numbers. Set $PROCESSING_RES=512 to reproduce the historical run.
$PROCESSING_RES = 768

# In/out-of-bbox abs_rel split for the surviving checkpoints of the first
# object_bbox_loss LoRA run (checkpoint-500..3000 were pruned by
# checkpoints_total_limit=10 during training).
$DETAIL_MODEL = "output/train-lotus-d-object-bbox-lora"
$REGRESSOR = "output/object_depth_regressor_v5_roi"
$ARTIFACTS = "D:/lotus/data/nyuv2_detail_artifacts/test"
$OUT_ROOT = "output/eval_bbox_split_object_bbox_lora_v1"

$STEPS = 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000

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
