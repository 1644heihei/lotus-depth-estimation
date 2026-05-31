export MODEL="jingheya/lotus-depth-d-v2-0-disparity"
export INPUT_DIR="assets/in-the-wild_example"
export OUTPUT_ROOT="output/semantic_ablation"

# Required for ROI metrics
export GT_DIR="$PATH_TO_GT_DEPTH_NPY"
export MASK_DIR="$PATH_TO_OBJECT_MASKS"
export CLASS_MAP_JSON="$PATH_TO_CLASS_MAP_JSON"

python experiments/run_semantic_depth_ablation.py \
  --model "$MODEL" \
  --input_dir "$INPUT_DIR" \
  --output_root "$OUTPUT_ROOT" \
  --mask_dir "$MASK_DIR" \
  --mode regression \
  --half_precision \
  --weights "0.3,0.5,0.7,1.0" \
  --gt_dir "$GT_DIR" \
  --class_map_json "$CLASS_MAP_JSON"
