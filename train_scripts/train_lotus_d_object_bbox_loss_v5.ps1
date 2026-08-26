$ErrorActionPreference = "Stop"

# Experiment: LR sweep continuation (docs/object_bbox_loss_investigation.md, improvement plan #1).
# v2 (LR=1e-4, constant) -> v3 (LR=1e-4, cosine) -> v4 (LR=3e-5, cosine) improved monotonically,
# but the best checkpoint was still step-250 in all three - i.e. still within warmup (500 steps),
# meaning the LR hadn't even reached its peak yet at the best point. This run drops peak LR
# further, 3e-5 -> 1e-5, keeping cosine + warmup=500, to see if the trend keeps improving.
$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed"
$OBJECT_BBOX_DETECTIONS_ROOT = "D:/lotus/data/hypersim_yolo_detections/train"
$OUTPUT_DIR = "output/train-lotus-d-object-bbox-lora-v5"

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
  --learning_rate=1e-5 `
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
