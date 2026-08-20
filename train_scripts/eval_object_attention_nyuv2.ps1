$ErrorActionPreference = "Stop"
Set-Location "D:/lotus/lotus-depth-estimation"

$REGRESSOR = "output/object_depth_regressor_v4_mask"
$ARTIFACTS = "D:/lotus/data/nyuv2_detail_artifacts/test"
$RGB = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
$OFFICIAL = "jingheya/lotus-depth-d-v2-0-disparity"
$CONTROL = "output/train-lotus-d-rgb-control-4ch"
$ATTENTION = "output/train-lotus-d-object-attention-4ch"

function Run-Eval {
    param($Model, $Mode, $Output)
    python eval_regressor_predepth_nyuv2.py `
      --detail_model=$Model `
      --regressor_dir=$REGRESSOR `
      --detail_artifacts_dir=$ARTIFACTS `
      --rgb_dir=$RGB `
      --condition_mode=$Mode `
      --max_objects=16 `
      --processing_res=512 `
      --half_precision `
      --min_detections=1 `
      --output_dir=$Output
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Run-Eval $OFFICIAL "unconditioned" "output/eval-object-attention/official"
Run-Eval $CONTROL "attention_off" "output/eval-object-attention/rgb-control"
Run-Eval $ATTENTION "attention_off" "output/eval-object-attention/attention-off"
Run-Eval $ATTENTION "attention" "output/eval-object-attention/attention-on"
