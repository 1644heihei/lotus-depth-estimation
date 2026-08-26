$ErrorActionPreference = "Stop"

# NOTE: processing_res was 512 in the original runs - a misconfiguration inherited from
# the regressor config (see docs/phase0_findings.md 3.3.1). 768 is Lotus's official
# default. Recorded results in docs/ were produced at 512; re-running now gives
# different (better) numbers. Set $PROCESSING_RES=512 to reproduce the historical run.
$PROCESSING_RES = 768

python utils/build_detail_train_dataset.py `
  --rgb_dir="D:/lotus/data/scannet_depth_eval_v1" `
  --output_dir="D:/lotus/data/scannet_crop_predepth_v1" `
  --detections_root="D:/lotus/data/scannet_yolo_artifacts_v1" `
  --core_model="jingheya/lotus-depth-d-v2-0-disparity" `
  --steps=predepth `
  --pattern="*.jpg" `
  --processing_res=$PROCESSING_RES `
  --align_mode=lstsq `
  --half_precision
