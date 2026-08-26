$ErrorActionPreference = "Stop"

Write-Host "=== Experiment v6 ablation (1/2): plain LoRA fine-tune, no object_bbox_loss (0-3000 steps) ==="
& "$PSScriptRoot\train_lotus_d_object_bbox_loss_v6_ablation.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 1 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== Experiment v6 ablation (2/2): bbox-split eval on v6 checkpoints (250-3000) ==="
& "$PSScriptRoot\eval_bbox_split_nyuv2_v6_ablation.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 2 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== Experiment v6 ablation done ==="
exit 0
