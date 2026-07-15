$ErrorActionPreference = "Stop"

# ========= User settings: Approach A offline dataset build =========
$RGB_DIR = "data/hypersim_processed/train"
$OUTPUT_DIR = "data/detail_train/train"
$CORE_MODEL = "output/train-lotus-d-core-bsz8/checkpoint-20000"
$MAX_IMAGES = 0   # 0 = all
$STEPS = "all"    # yolo,predepth,classmap,all

# =================================

Write-Host "Building Approach-A detail training dataset"
Write-Host "RGB_DIR: $RGB_DIR"
Write-Host "OUTPUT_DIR: $OUTPUT_DIR"
Write-Host "CORE_MODEL: $CORE_MODEL"

python utils/build_detail_train_dataset.py `
  --rgb_dir=$RGB_DIR `
  --output_dir=$OUTPUT_DIR `
  --core_model=$CORE_MODEL `
  --steps=$STEPS `
  --pattern="rgb_cam_*.png" `
  --max_images=$MAX_IMAGES `
  --skip_existing `
  --half_precision
