export CUDA=0

export CHECKPOINT_DIR="jingheya/lotus-depth-d-v2-0-disparity"
export OUTPUT_DIR="output/Depth_D_ROI_Fusion"
export INPUT_DIR="assets/in-the-wild_example"

# Optional: generated masks from YOLO/SAM (same basename)
export ROI_MASK_DIR="$PATH_TO_ROI_MASKS"

CUDA_VISIBLE_DEVICES=$CUDA python infer_roi_fusion.py \
  --pretrained_model_name_or_path=$CHECKPOINT_DIR \
  --mode="regression" \
  --task_name="depth" \
  --input_dir=$INPUT_DIR \
  --output_dir=$OUTPUT_DIR \
  --half_precision \
  --timestep=999 \
  --seed=42 \
  --roi_mask_dir=$ROI_MASK_DIR \
  --fusion_weight=1.0 \
  --roi_expand_ratio=0.12 \
  --roi_min_area=500 \
  --blend_blur_ksize=31
