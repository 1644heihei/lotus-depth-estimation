$ErrorActionPreference = "Stop"

# Contour conditioning: feed a SAM contour to the UNet as extra input channels.
#
# Post-processing with a contour is worth +279.9% BF1 net of control, with no training and
# abs_rel improving - the largest ceiling in this investigation. It is out of reach that
# way because the operation trusts the contour's position and needs 1.71px, while SAM
# delivers 2.00px. Training is what could learn to read a contour as "a step is near here"
# rather than "the step is exactly here", and that is the only thing post-processing
# cannot do.
#
# Two measurements say the input-channel route can carry it. A contour survives Lotus's
# VAE nearly intact - 91.1% of its pixels return exactly, 99.8% within the 1.71px budget -
# so the 8x downsample does not destroy the precision the channel exists for. And
# expand_unet_conv_in zero-initialises the appended slices, so step 0 reproduces the
# pretrained model instead of opening with the tax every LoRA run here has paid.
#
#   C-null   an empty contour channel   does the zero-initialised expansion cost anything
#            by itself? This is the baseline the other two are read against, and the
#            reason it exists is that LoRA on cross-attention cost 37.4% of BF1 while
#            conditioning on nothing at all.
#   C-cont   this image's contour       the treatment
#   C-shuf   a neighbour's contour      the same amount of contour describing the wrong
#            scene. Separates the position information from the channel merely being full.
#
# Matched across all three: rank 8 / alpha 16 / LR 1e-5 cosine / warmup 200 / batch 8 /
# res 512 / 3000 steps / seed 42, so any difference is the contour.

$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed"
$CONTOURS = "D:/lotus/data/hypersim_sam_seg"
$HOLDOUT = "datasets/hypersim_holdout.json"
$STEPS = 3000

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"

# name, contour mode
$CONFIGS = @(
  @("null", "zero"),
  @("cont", "real"),
  @("shuf", "shuffled")
)

foreach ($c in $CONFIGS) {
  $name, $mode = $c
  $out = "output/lora-contour-$name"
  if (Test-Path "$out/unet_lora") {
    Write-Host "=== skip C-$name (already trained) ==="
    continue
  }
  Write-Host "=== training C-$name  contour_mode=$mode ==="

  accelerate launch `
    --config_file="accelerate_configs/0.yaml" `
    --mixed_precision="bf16" `
    --main_process_port=13342 `
    train_lotus_d.py `
    --pretrained_model_name_or_path=$BASE_MODEL `
    --train_data_dir_hypersim=$RGB_ROOT `
    --use_lora `
    --lora_target_blocks="all" `
    --lora_rank=8 `
    --lora_alpha=16 `
    --contour_condition `
    --contour_mask_root=$CONTOURS `
    --contour_mode=$mode `
    --contour_width=1 `
    --hypersim_holdout_split=$HOLDOUT `
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
    --checkpoints_total_limit=8 `
    --base_test_data_dir="datasets/eval/" `
    --output_dir=$out `
    --grad_loss_weight=0.1 `
    --disable_rgb_reconstruction `
    --resume_from_checkpoint="latest"

  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: C-$name"; exit $LASTEXITCODE }
  Write-Host "=== done C-$name ==="
}

Write-Host "=== all three contour runs finished ==="
exit 0
