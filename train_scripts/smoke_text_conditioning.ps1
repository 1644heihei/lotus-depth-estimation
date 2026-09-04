$ErrorActionPreference = "Stop"

# 20-step smoke test for the class-name text conditioning (option B + C).
#
# Not a training run - it exists to prove the configuration actually reaches the loop
# before three 67-minute runs are launched on it. Two previous experiments in this repo
# were reported as started when they had not started at all (execution policy blocking a
# nested launch; "--lora_alpha=($rank * 2)" splitting into two arguments), so the rule
# is that nothing counts as running until the log shows the intended settings.
#
# What the log must show:
#   LoRA blocks=text  trainable params=0.86M      cross-attention only, not zero
#   Class-name prompts ON ... spatial_bias=True   the flags took effect
#   Hypersim hold-out: 20 scenes excluded         the split is applied
#   [prompt sample] ...                           per-sample prompts differ

$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed"
# note the doubled train/: the JSONs sit under <root>/train/train/ai_XXX_YYY/
$DETECTIONS = "D:/lotus/data/hypersim_yolo_detections/train"
$HOLDOUT = "datasets/hypersim_holdout.json"

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"

accelerate launch `
  --config_file="accelerate_configs/0.yaml" `
  --mixed_precision="bf16" `
  --main_process_port=13337 `
  train_lotus_d.py `
  --pretrained_model_name_or_path=$BASE_MODEL `
  --train_data_dir_hypersim=$RGB_ROOT `
  --use_lora `
  --lora_target_blocks="text" `
  --lora_rank=8 `
  --lora_alpha=16 `
  --class_name_prompts `
  --class_name_detections_root=$DETECTIONS `
  --text_prompt_dropout_p=0.1 `
  --class_token_spatial_bias `
  --class_token_bias_dropout_p=0.5 `
  --hypersim_holdout_split=$HOLDOUT `
  --resolution_hypersim=512 `
  --norm_type="trunc_disparity" `
  --dataloader_num_workers=0 `
  --train_batch_size=8 `
  --gradient_accumulation_steps=1 `
  --gradient_checkpointing `
  --max_grad_norm=1 `
  --seed=42 `
  --max_train_steps=20 `
  --learning_rate=1e-5 `
  --lr_scheduler="cosine" `
  --lr_warmup_steps=200 `
  --task_name="depth" `
  --timestep=999 `
  --validation_images="datasets/quick_validation/" `
  --validation_steps=100000 `
  --checkpointing_steps=100000 `
  --base_test_data_dir="datasets/eval/" `
  --output_dir="output/_smoke-text-conditioning" `
  --grad_loss_weight=0.1 `
  --disable_rgb_reconstruction

if ($LASTEXITCODE -ne 0) { Write-Host "SMOKE FAILED"; exit $LASTEXITCODE }
Write-Host "=== smoke test finished ==="
exit 0
