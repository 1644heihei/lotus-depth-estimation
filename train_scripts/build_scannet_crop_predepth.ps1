$ErrorActionPreference = "Stop"

python utils/build_detail_train_dataset.py `
  --rgb_dir="D:/lotus/data/scannet_depth_eval_v1" `
  --output_dir="D:/lotus/data/scannet_crop_predepth_v1" `
  --detections_root="D:/lotus/data/scannet_yolo_artifacts_v1" `
  --core_model="jingheya/lotus-depth-d-v2-0-disparity" `
  --steps=predepth `
  --pattern="*.jpg" `
  --processing_res=512 `
  --align_mode=lstsq `
  --half_precision
