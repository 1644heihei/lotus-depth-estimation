$ErrorActionPreference = "Stop"

# Conditioned NYUv2 evaluation for Approach-A detail models.
# Requires offline artifacts: D:/lotus/data/nyuv2_detail_artifacts/test

$WorkDir = "D:\lotus\lotus-depth-estimation"
Set-Location $WorkDir

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:HUGGING_FACE_HUB_TOKEN -ErrorAction SilentlyContinue

# Default: 12ch best mid ckpt (step 6000) assembled pipeline
$DETAIL_MODEL = if ($env:DETAIL_MODEL) { $env:DETAIL_MODEL } else { "D:/lotus/data/tmp_pipe_12ch_step6000" }
$ARTIFACTS = "D:/lotus/data/nyuv2_detail_artifacts/test"
$OUT = if ($env:EVAL_OUT) { $env:EVAL_OUT } else { "output/eval_nyuv2_conditioned_12ch_step6000" }
$SCORE = if ($env:DET_SCORE) { $env:DET_SCORE } else { "0.5" }
$MAX = if ($env:MAX_IMAGES) { $env:MAX_IMAGES } else { "0" }

Write-Host "DETAIL_MODEL=$DETAIL_MODEL"
Write-Host "ARTIFACTS=$ARTIFACTS"
Write-Host "OUT=$OUT"

if (-not (Test-Path $ARTIFACTS)) {
  throw "Artifacts dir missing: $ARTIFACTS  (run train_scripts/build_nyuv2_detail_artifacts.ps1 first)"
}

$detCount = (Get-ChildItem $ARTIFACTS -Recurse -Filter "*_detections.json" -File -EA SilentlyContinue | Measure-Object).Count
Write-Host "detections_json_count=$detCount"
if ($detCount -lt 654) {
  Write-Host "WARNING: expected 654 detections, found $detCount (build may still be running)"
}

python eval_detail_conditioned_nyuv2.py `
  --detail_model=$DETAIL_MODEL `
  --detail_artifacts_dir=$ARTIFACTS `
  --output_dir=$OUT `
  --detection_score_thr=$SCORE `
  --half_precision `
  --max_images=$MAX
