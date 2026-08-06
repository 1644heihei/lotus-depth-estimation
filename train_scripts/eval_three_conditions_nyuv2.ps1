$ErrorActionPreference = "Stop"

$MODEL = "output/train-lotus-d-regressor-predepth-9ch"
$REGRESSOR = "output/object_depth_regressor_v3"
$CROP_PRE_DEPTH = "D:/lotus/data/nyuv2_detail_artifacts/test"
$OUT = "output/eval_three_conditions_nyuv2"
$COMMON = @(
  "--detail_model=$MODEL",
  "--regressor_dir=$REGRESSOR",
  "--processing_res=512",
  "--detection_score_thr=0.5",
  "--seed=42",
  "--half_precision"
)

python eval_regressor_predepth_nyuv2.py @COMMON `
  --condition_mode=unconditioned `
  --output_dir="$OUT/unconditioned"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python eval_regressor_predepth_nyuv2.py @COMMON `
  --condition_mode=cached `
  --pre_depth_root=$CROP_PRE_DEPTH `
  --output_dir="$OUT/crop_predepth"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

python eval_regressor_predepth_nyuv2.py @COMMON `
  --condition_mode=regressor `
  --output_dir="$OUT/regressor_predepth"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
