$ErrorActionPreference = "Stop"

# Evaluation for the three text-conditioning runs, against the pre-registered criteria in
# docs/text_conditioning_training_plan.md section 5.
#
# Four model settings, because T-both has to be read both ways:
#   T-null            the fine-tune tax at matched capacity
#   T-text            option B alone
#   T-both, bias ON   the configuration that would actually be deployed
#   T-both, bias OFF  what B learned underneath C - criterion (d) is measured here, since
#                     with the bias on the lift would just be the bias
#
# Prompt variants stop at four: classes_pos / classes_wrongpos tested option A, which the
# plan rejected after right and wrong positions came out indistinguishable, and dropping
# them saves 654 forward passes per run.
#
# Baselines to beat (untrained Lotus, NYUv2 654):
#   abs_rel 0.05000 empty / 0.05180 classes    recall OFF edge 23.9% empty / 27.3% classes
#   cross-attention lift 1.079 (chance 1.00, perfect binding 5.18)

$env:PYTHONIOENCODING = "utf-8"
$NYU = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"

# tag, lora dir, bias flag
$RUNS = @(
  @("null",     "output/lora-text-null/unet_lora", $false),
  @("text",     "output/lora-text-text/unet_lora", $false),
  @("both_on",  "output/lora-text-both/unet_lora", $true),
  @("both_off", "output/lora-text-both/unet_lora", $false)
)

foreach ($r in $RUNS) {
  $tag, $lora, $bias = $r
  if (-not (Test-Path $lora)) {
    Write-Host "=== skip $tag (no adapter at $lora) ==="
    continue
  }
  Write-Host "=== evaluating $tag  bias=$bias ==="

  $extra = @()
  if ($bias) { $extra += "--class_token_spatial_bias" }

  python -u eval_text_prompt_conditioning.py `
    --rgb_dir=$NYU `
    --lora_path=$lora `
    --run_tag=$tag `
    --variants empty classes shuffled generic `
    --half_precision `
    --output_dir="output/eval_text_$tag" `
    @extra
  if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: eval $tag"; exit $LASTEXITCODE }

  # Criterion (d) only. Always bias OFF: with it on the lift measures the bias, not
  # whether training grew the binding.
  if (-not $bias) {
    python -u eval_cross_attention_localization.py `
      --lora_path=$lora `
      --run_tag=$tag `
      --max_images=250 `
      --output_dir="output/eval_xattn_$tag"
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: xattn $tag"; exit $LASTEXITCODE }
  }
}

Write-Host "=== all evaluations finished ==="
exit 0
