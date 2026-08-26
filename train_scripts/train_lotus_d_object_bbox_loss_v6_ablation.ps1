$ErrorActionPreference = "Stop"

# Ablation: identical to v4 (LR=3e-5, cosine, warmup=500, 3000 steps) but WITHOUT
# --object_bbox_detections_root, so no object_bbox_loss term is added at all - this is a
# plain LoRA fine-tune on the same data/seed/schedule. v2-v5 always had object_bbox_loss_weight=1.0
# on, so we never isolated whether the bbox loss term itself contributes anything beyond what
# LoRA fine-tuning does on its own. This run is the missing control.
$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed"
$OUTPUT_DIR = "output/train-lotus-d-object-bbox-lora-v6-ablation"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

accelerate launch `
  --config_file="accelerate_configs/0.yaml" `
  --mixed_precision="bf16" `
  --main_process_port=13329 `
  train_lotus_d.py `
  --pretrained_model_name_or_path=$BASE_MODEL `
  --train_data_dir_hypersim=$RGB_ROOT `
  --use_lora `
  --lora_rank=8 `
  --lora_alpha=16 `
  --resolution_hypersim=512 `
  --norm_type="trunc_disparity" `
  --dataloader_num_workers=0 `
  --train_batch_size=8 `
  --gradient_accumulation_steps=1 `
  --gradient_checkpointing `
  --max_grad_norm=1 `
  --seed=42 `
  --max_train_steps=3000 `
  --learning_rate=3e-5 `
  --lr_scheduler="cosine" `
  --lr_warmup_steps=500 `
  --task_name="depth" `
  --timestep=999 `
  --validation_images="datasets/quick_validation/" `
  --validation_steps=250 `
  --checkpointing_steps=250 `
  --checkpoints_total_limit=20 `
  --base_test_data_dir="datasets/eval/" `
  --output_dir=$OUTPUT_DIR `
  --grad_loss_weight=0.1 `
  --disable_rgb_reconstruction `
  --resume_from_checkpoint="latest"
exit $LASTEXITCODE
