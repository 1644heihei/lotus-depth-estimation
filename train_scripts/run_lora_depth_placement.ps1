$ErrorActionPreference = "Stop"

# Gate for the diagnosis-driven adaptation idea (docs/global_error_findings.md):
# does adapting a given UNet depth move the matching error component?
#
# UNet depth corresponds to spatial scale - mid_block runs at the coarsest latent
# resolution, down_blocks.0 / up_blocks.3 at the finest. If the error decomposition is
# actionable, LoRA on the coarse blocks should shift the global component and LoRA on
# the fine blocks the local one. If both move everything the same way, the idea dies
# here and costs half a day instead of a thesis.
#
# LoRA parameter counts are matched between global and local (0.958M vs 0.946M) by
# raising local's rank, so any difference is placement rather than capacity.

$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed"
$STEPS = 1000

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

# name, target blocks, rank
$CONFIGS = @(
  @("global", "global", 8),
  @("local",  "local",  29),
  @("all",    "all",    8)
)

foreach ($c in $CONFIGS) {
  $name, $blocks, $rank = $c
  # compute into a variable: "--lora_alpha=($rank * 2)" splits into two arguments
  $alpha = $rank * 2
  $out = "output/lora-placement-$name"
  if (Test-Path "$out/unet_lora") {
    Write-Host "=== skip $name (already trained) ==="
    continue
  }
  Write-Host "=== training: blocks=$blocks rank=$rank ==="
  accelerate launch `
    --config_file="accelerate_configs/0.yaml" `
    --mixed_precision="bf16" `
    --main_process_port=13331 `
    train_lotus_d.py `
    --pretrained_model_name_or_path=$BASE_MODEL `
    --train_data_dir_hypersim=$RGB_ROOT `
    --use_lora `
    --lora_target_blocks=$blocks `
    --lora_rank=$rank `
    --lora_alpha=$alpha `
    --resolution_hypersim=512 `
    --norm_type="trunc_disparity" `
    --dataloader_num_workers=0 `
    --train_batch_size=8 `
    --gradient_accumulation_steps=1 `
    --gradient_checkpointing `
    --max_grad_norm=1 `
    --seed=42 `
    --max_train_steps=$STEPS `
    --learning_rate=1e-5 `
    --lr_scheduler="cosine" `
    --lr_warmup_steps=200 `
    --task_name="depth" `
    --timestep=999 `
    --validation_images="datasets/quick_validation/" `
    --validation_steps=100000 `
    --checkpointing_steps=500 `
    --checkpoints_total_limit=5 `
    --base_test_data_dir="datasets/eval/" `
    --output_dir=$out `
    --grad_loss_weight=0.1 `
    --disable_rgb_reconstruction `
    --resume_from_checkpoint="latest"
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $name"; exit $LASTEXITCODE }
}

Write-Host "=== all placements trained ==="
exit 0
