$ErrorActionPreference = "Stop"

# Experiment: LR schedule change (docs/object_bbox_loss_investigation.md, "improvement plan" #1).
# v1/v2 both used lr_scheduler=constant and peaked at a very early checkpoint (step-250/3500)
# then degraded steadily - a classic overfitting signature. This run switches to cosine decay
# (peak LR unchanged at 1e-4, same warmup/steps/loss weights as v2) to isolate the scheduler's
# effect from everything else.
$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed"
$OBJECT_BBOX_DETECTIONS_ROOT = "D:/lotus/data/hypersim_yolo_detections/train"
$OUTPUT_DIR = "output/train-lotus-d-object-bbox-lora-v3"

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
  --object_bbox_detections_root=$OBJECT_BBOX_DETECTIONS_ROOT `
  --object_bbox_score_thr=0.5 `
  --object_bbox_loss_weight=1.0 `
  --resolution_hypersim=512 `
  --norm_type="trunc_disparity" `
  --dataloader_num_workers=0 `
  --train_batch_size=8 `
  --gradient_accumulation_steps=1 `
  --gradient_checkpointing `
  --max_grad_norm=1 `
  --seed=42 `
  --max_train_steps=3000 `
  --learning_rate=1e-4 `
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
