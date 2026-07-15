$ErrorActionPreference = "Stop"

# ========= User settings: Core model (Training 1) =========
$MODEL_NAME = "stabilityai/stable-diffusion-2-base"
$TRAIN_DATA_DIR_HYPERSIM = "hf://omrastogi/Hypersim-Processed/train?cache_dir=data/hf_cache/hypersim_processed"
$TRAIN_DATA_DIR_VKITTI = ""
$RES_HYPERSIM = 512
$P_HYPERSIM = 1.0
$NORMTYPE = "trunc_disparity"

$BATCH_SIZE = 8
$CUDA = "0"
$GAS = 1
$TIMESTEP = 999
$TASK_NAME = "depth"
$VAL_STEP = 500
$MAX_TRAIN_STEPS = 20000
$LR = 3e-5
$MIXED_PRECISION = "bf16"

$BASE_TEST_DATA_DIR = "datasets/eval/"
$VALIDATION_IMAGES = "datasets/quick_validation/"
$OUTPUT_DIR = "output/train-lotus-d-core-bsz$BATCH_SIZE/"

# =================================

Write-Host "Starting Training 1: Core depth model"
Write-Host "MODEL_NAME: $MODEL_NAME"
Write-Host "OUTPUT_DIR: $OUTPUT_DIR"

accelerate launch `
  --config_file="accelerate_configs/$CUDA.yaml" `
  --mixed_precision=$MIXED_PRECISION `
  --main_process_port=13324 `
  train_lotus_d.py `
  --pretrained_model_name_or_path=$MODEL_NAME `
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM `
  --resolution_hypersim=$RES_HYPERSIM `
  --prob_hypersim=$P_HYPERSIM `
  --random_flip `
  --norm_type=$NORMTYPE `
  --dataloader_num_workers=0 `
  --train_batch_size=$BATCH_SIZE `
  --gradient_accumulation_steps=$GAS `
  --gradient_checkpointing `
  --max_grad_norm=1 `
  --seed=42 `
  --max_train_steps=$MAX_TRAIN_STEPS `
  --learning_rate=$LR `
  --lr_scheduler="constant" --lr_warmup_steps=0 `
  --task_name=$TASK_NAME `
  --timestep=$TIMESTEP `
  --validation_images=$VALIDATION_IMAGES `
  --validation_steps=$VAL_STEP `
  --checkpointing_steps=$VAL_STEP `
  --base_test_data_dir=$BASE_TEST_DATA_DIR `
  --output_dir=$OUTPUT_DIR `
  --resume_from_checkpoint="latest"
