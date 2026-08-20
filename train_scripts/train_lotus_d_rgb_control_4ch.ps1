$ErrorActionPreference = "Stop"

# Matched control: same images and optimizer schedule as object-attention training,
# but object tokens are dropped for every sample.
$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed/train"
$YOLO_ROOT = "D:/lotus/data/hypersim_yolo_detections/train"
$CACHE = "D:/lotus/data/object_attention_cache_v1/hypersim_train.npz"
$MANIFEST = "D:/lotus/data/hypersim_yolo_detections/train_manifest_score0.5.json"
$OUTPUT_DIR = "output/train-lotus-d-rgb-control-4ch"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

accelerate launch `
  --config_file="accelerate_configs/0.yaml" `
  --mixed_precision="bf16" `
  --main_process_port=13330 `
  train_lotus_d.py `
  --pretrained_model_name_or_path=$BASE_MODEL `
  --train_data_dir_hypersim=$RGB_ROOT `
  --detail_train_data_dir=$YOLO_ROOT `
  --detail_train_manifest=$MANIFEST `
  --object_attention_cache=$CACHE `
  --enable_object_attention `
  --max_objects=16 `
  --object_attention_dropout_p=1.0 `
  --object_encoder_lr=1e-4 `
  --resolution_hypersim=512 `
  --norm_type="trunc_disparity" `
  --dataloader_num_workers=0 `
  --train_batch_size=8 `
  --gradient_accumulation_steps=1 `
  --gradient_checkpointing `
  --max_grad_norm=1 `
  --seed=42 `
  --max_train_steps=12000 `
  --learning_rate=1e-5 `
  --lr_scheduler="constant" `
  --lr_warmup_steps=0 `
  --task_name="depth" `
  --timestep=999 `
  --validation_images="datasets/quick_validation/" `
  --validation_steps=500 `
  --checkpointing_steps=500 `
  --checkpoints_total_limit=2 `
  --base_test_data_dir="datasets/eval/" `
  --output_dir=$OUTPUT_DIR `
  --detection_score_thr=0.5 `
  --grad_loss_weight=0.1 `
  --disable_rgb_reconstruction
exit $LASTEXITCODE
