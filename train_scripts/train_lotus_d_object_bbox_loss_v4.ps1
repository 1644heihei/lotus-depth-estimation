$ErrorActionPreference = "Stop"

# Experiment: lower peak LR (docs/object_bbox_loss_investigation.md, improvement plan #1,
# second half). v3 (cosine, LR=1e-4) confirmed cosine decay reduces late-run degradation but
# the best checkpoint was still step-250 - i.e. the model overfits almost immediately even
# during warmup. This run keeps cosine but drops peak LR 1e-4 -> 3e-5 to test whether a lower
# LR lets training continue improving past step-250 instead of overfitting right away.
$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed"
$OBJECT_BBOX_DETECTIONS_ROOT = "D:/lotus/data/hypersim_yolo_detections/train"
$OUTPUT_DIR = "output/train-lotus-d-object-bbox-lora-v4"

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
