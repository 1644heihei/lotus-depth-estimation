$ErrorActionPreference = "Stop"

$MODEL = "output/object_depth_regressor"
$DETAIL = "output/train-lotus-d-detail-bsz8"
$REGRESSOR = "output/object_depth_regressor"
$OUT = "output/eval_regressor_predepth_nyuv2"

Write-Host "NYUv2 eval: regressor pre-depth + 9ch detail model"

python eval_regressor_predepth_nyuv2.py `
  --detail_model=$DETAIL `
  --regressor_dir=$REGRESSOR `
  --output_dir=$OUT `
  --half_precision
