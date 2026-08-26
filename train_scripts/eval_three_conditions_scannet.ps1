$ErrorActionPreference = "Stop"

# NOTE: processing_res was 512 in the original runs - a misconfiguration inherited from
# the regressor config (see docs/phase0_findings.md 3.3.1). 768 is Lotus's official
# default. Recorded results in docs/ were produced at 512; re-running now gives
# different (better) numbers. Set $PROCESSING_RES=512 to reproduce the historical run.
$PROCESSING_RES = 768

$MODEL = "output/train-lotus-d-regressor-predepth-9ch"
$REGRESSOR = "output/object_depth_regressor_v3"
$RGB_ROOT = "D:/lotus/data/scannet_depth_eval_v1"
$YOLO_ROOT = "D:/lotus/data/scannet_yolo_artifacts_v1"
$CROP_PRE_DEPTH = "D:/lotus/data/scannet_crop_predepth_v1"
$OUT = "output/eval_three_conditions_scannet"
$COMMON = @(
  "--dataset=scannet",
  "--rgb_dir=$RGB_ROOT",
  "--detail_artifacts_dir=$YOLO_ROOT",
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

python eval_regressor_predepth_nyuv2.py @COMMON `
  --condition_mode=cached `
  --pre_depth_root=$CROP_PRE_DEPTH `
  --output_dir="$OUT/crop_predepth"

python eval_regressor_predepth_nyuv2.py @COMMON `
  --condition_mode=regressor `
  --output_dir="$OUT/regressor_predepth"
