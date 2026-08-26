$ErrorActionPreference = "Stop"

# NOTE: processing_res was 512 in the original runs - a misconfiguration inherited from
# the regressor config (see docs/phase0_findings.md 3.3.1). 768 is Lotus's official
# default. Recorded results in docs/ were produced at 512; re-running now gives
# different (better) numbers. Set $PROCESSING_RES=512 to reproduce the historical run.
$PROCESSING_RES = 768

$MODEL = "output/train-lotus-d-regressor-predepth-9ch"
$REGRESSOR = "output/object_depth_regressor_v3"
$CROP_PRE_DEPTH = "D:/lotus/data/nyuv2_detail_artifacts/test"
$OUT = "output/eval_three_conditions_nyuv2"
$COMMON = @(
  "--detail_model=$MODEL",
  "--regressor_dir=$REGRESSOR",
  "--processing_res=$PROCESSING_RES",
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
