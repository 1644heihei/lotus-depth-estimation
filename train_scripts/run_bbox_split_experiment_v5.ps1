$ErrorActionPreference = "Stop"

Write-Host "=== Experiment v5 (1/2): lower peak LR (1e-5, cosine) retrain (0-3000 steps) ==="
& "$PSScriptRoot\train_lotus_d_object_bbox_loss_v5.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 1 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== Experiment v5 (2/2): bbox-split eval on v5 checkpoints (250-3000) ==="
& "$PSScriptRoot\eval_bbox_split_nyuv2_v5.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 2 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== Experiment v5 done ==="
exit 0
