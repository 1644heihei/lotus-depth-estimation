$ErrorActionPreference = "Stop"

# ========= Detail model 12ch (direct class + size_w/h, no VAE on object maps) =========
# Channels: RGB(4) + pre(4) + valid(1) + class(1) + size_w(1) + size_h(1) = 12
# Init from official Lotus-D + zero-init expand.
$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_TOKEN = $null
$env:HUGGING_FACE_HUB_TOKEN = $null
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:HUGGING_FACE_HUB_TOKEN -ErrorAction SilentlyContinue

$TRAIN_DATA_DIR_HYPERSIM = "D:/lotus/data/hypersim_processed/train"
$DETAIL_TRAIN_DATA_DIR = "D:/lotus/data/hypersim_yolo_detections/train"
$RES_HYPERSIM = 512
$NORMTYPE = "trunc_disparity"

$BATCH_SIZE = 8
$CUDA = "0"
$GAS = 1
$TIMESTEP = 999
$TASK_NAME = "depth"
$VAL_STEP = 500
$MAX_TRAIN_STEPS = 12000
$LR = 3e-5
$MIXED_PRECISION = "bf16"

$PRE_DEPTH_DROPOUT = 0.3
$DETECTION_SCORE_THR = 0.5
$GRAD_LOSS_WEIGHT = 0.1

$BASE_TEST_DATA_DIR = "datasets/eval/"
$VALIDATION_IMAGES = "datasets/quick_validation/"
$OUTPUT_DIR = "output/train-lotus-d-detail-12ch-bsz$BATCH_SIZE/"

Write-Host "Starting Detail depth model (12ch: class+size direct)"
Write-Host "BASE_MODEL: $BASE_MODEL"
Write-Host "OUTPUT_DIR: $OUTPUT_DIR"

accelerate launch `
  --config_file="accelerate_configs/$CUDA.yaml" `
  --mixed_precision=$MIXED_PRECISION `
  --main_process_port=13326 `
  train_lotus_d.py `
  --pretrained_model_name_or_path=$BASE_MODEL `
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM `
  --detail_train_data_dir=$DETAIL_TRAIN_DATA_DIR `
  --resolution_hypersim=$RES_HYPERSIM `
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
  --enable_object_condition `
  --enable_bbox_size_condition `
  --pre_depth_dropout_p=$PRE_DEPTH_DROPOUT `
  --detection_score_thr=$DETECTION_SCORE_THR `
  --grad_loss_weight=$GRAD_LOSS_WEIGHT `
  --disable_rgb_reconstruction
