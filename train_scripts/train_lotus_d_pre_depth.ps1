$ErrorActionPreference = "Stop"

# ========= User settings =========
$MODEL_NAME = "jingheya/lotus-depth-d-v2-0-disparity"
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

# Phase-2 knobs
$PRE_DEPTH_DROPOUT = 0.3
$PRE_DEPTH_SCALE_JITTER = 0.10
$PRE_DEPTH_SHIFT_JITTER = 0.05
$PRE_DEPTH_BLUR_SIGMA_MAX = 1.5
$PRE_DEPTH_HOLE_KEEP = 0.92
$PRE_DEPTH_NOISE_STD = 0.01
$GRAD_LOSS_WEIGHT = 0.1

$BASE_TEST_DATA_DIR = "datasets/eval/"
$VALIDATION_IMAGES = "datasets/quick_validation/"
$OUTPUT_DIR = "output/train-lotus-d-pre-depth-bsz$BATCH_SIZE/"

# =================================

Write-Host "Starting Phase-2 pre-depth training"
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
  --resume_from_checkpoint="latest" `
  --enable_pre_depth_fusion `
  --pre_depth_dropout_p=$PRE_DEPTH_DROPOUT `
  --pre_depth_scale_jitter=$PRE_DEPTH_SCALE_JITTER `
  --pre_depth_shift_jitter=$PRE_DEPTH_SHIFT_JITTER `
  --pre_depth_blur_sigma_max=$PRE_DEPTH_BLUR_SIGMA_MAX `
  --pre_depth_hole_keep_p=$PRE_DEPTH_HOLE_KEEP `
  --pre_depth_noise_std=$PRE_DEPTH_NOISE_STD `
  --grad_loss_weight=$GRAD_LOSS_WEIGHT
