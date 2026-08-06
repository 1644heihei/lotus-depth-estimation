$ErrorActionPreference = "Stop"

# Dedicated 9ch model: RGB latent(4) + pre-depth latent(4) + valid mask(1).
# Always initializes from the official 4ch Lotus-D model.
$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed/train"
$YOLO_ROOT = "D:/lotus/data/hypersim_yolo_detections/train"
$PRE_DEPTH_ROOT = "D:/lotus/data/regressor_predepth_v4_512_compact/hypersim_train"
$OUTPUT_DIR = "output/train-lotus-d-regressor-predepth-9ch"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

accelerate launch `
  --config_file="accelerate_configs/0.yaml" `
  --mixed_precision="bf16" `
  --main_process_port=13327 `
  train_lotus_d.py `
  --pretrained_model_name_or_path=$BASE_MODEL `
  --train_data_dir_hypersim=$RGB_ROOT `
  --detail_train_data_dir=$YOLO_ROOT `
  --pre_depth_artifacts_dir=$PRE_DEPTH_ROOT `
  --resolution_hypersim=512 `
  --norm_type="trunc_disparity" `
  --dataloader_num_workers=0 `
  --train_batch_size=8 `
  --gradient_accumulation_steps=1 `
  --gradient_checkpointing `
  --max_grad_norm=1 `
  --seed=42 `
  --max_train_steps=12000 `
  --learning_rate=3e-5 `
  --lr_scheduler="constant" `
  --lr_warmup_steps=0 `
  --task_name="depth" `
  --timestep=999 `
  --validation_images="datasets/quick_validation/" `
  --validation_steps=500 `
  --checkpointing_steps=500 `
  --base_test_data_dir="datasets/eval/" `
  --output_dir=$OUTPUT_DIR `
  --enable_pre_depth_fusion `
  --pre_depth_dropout_p=0.1 `
  --detection_score_thr=0.5 `
  --grad_loss_weight=0.1 `
  --disable_rgb_reconstruction
