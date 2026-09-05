$ErrorActionPreference = "Stop"

# The three runs of docs/text_conditioning_training_plan.md, section 3.
#
# Lotus recovers 31.2% of true depth discontinuities that sit on an image edge but only
# 10.5% of those in visually flat regions, and flat regions hold 54.6% of them. A chair
# the same colour as the wall behind it has a depth step with nothing in the image to mark
# it; a human still knows it is there because they know it is a chair. Lotus carries
# Stable Diffusion's CLIP encoder for exactly that kind of knowledge and was fine-tuned
# with prompt="" throughout, which flattened the text-to-region binding to 1.9% of full
# strength. Untrained, class names move the output no more usefully than the fixed string
# "an indoor scene" does, so training is the only version of this that could work.
#
#   T-null   empty prompts          isolates the fine-tune tax at matched capacity
#   T-text   class names            option B, binding rebuilt by training alone
#   T-both   class names + bias     option C hands the location over as well
#
# T-null runs first on purpose: without it, neither of the others can be read, because
# every conditioning experiment in this repo has paid a tax and T-text alone cannot say
# whether text helped or the LoRA did.
#
# Matched across all three: rank 8 / alpha 16 / LR 1e-5 cosine / warmup 200 / batch 8 /
# res 512 / 3000 steps / seed 42, so any difference is the conditioning.

$BASE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"
$RGB_ROOT = "D:/lotus/data/hypersim_processed"
# note the doubled train/: the JSONs sit under <root>/train/train/ai_XXX_YYY/
$DETECTIONS = "D:/lotus/data/hypersim_yolo_detections/train"
$HOLDOUT = "datasets/hypersim_holdout.json"
$STEPS = 3000

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONIOENCODING = "utf-8"

# name, class-name prompts, spatial bias
$CONFIGS = @(
  @("null", $false, $false),
  @("text", $true,  $false),
  @("both", $true,  $true)
)

foreach ($c in $CONFIGS) {
  $name, $useText, $useBias = $c
  $out = "output/lora-text-$name"
  if (Test-Path "$out/unet_lora") {
    Write-Host "=== skip T-$name (already trained) ==="
    continue
  }
  Write-Host "=== training T-$name  prompts=$useText  bias=$useBias ==="

  # built as an array: "--flag=$expr" is evaluated before splatting, and a previous
  # experiment lost a run to "--lora_alpha=($rank * 2)" splitting into two arguments
  $extra = @()
  if ($useText) {
    $extra += "--class_name_prompts"
    $extra += "--class_name_detections_root=$DETECTIONS"
    $extra += "--text_prompt_dropout_p=0.1"
  }
  if ($useBias) {
    $extra += "--class_token_spatial_bias"
    $extra += "--class_token_bias_dropout_p=0.5"
  }

  accelerate launch `
    --config_file="accelerate_configs/0.yaml" `
    --mixed_precision="bf16" `
    --main_process_port=13338 `
    train_lotus_d.py `
    --pretrained_model_name_or_path=$BASE_MODEL `
    --train_data_dir_hypersim=$RGB_ROOT `
    --use_lora `
    --lora_target_blocks="text" `
    --lora_rank=8 `
    --lora_alpha=16 `
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
    --resume_from_checkpoint="latest" `
    @extra

  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: T-$name"; exit $LASTEXITCODE }
  Write-Host "=== done T-$name ==="
}

Write-Host "=== all three runs finished ==="
exit 0
