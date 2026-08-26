$ErrorActionPreference = "Stop"

Write-Host "=== Experiment v3 (1/2): cosine LR retrain (0-3000 steps) ==="
& "$PSScriptRoot\train_lotus_d_object_bbox_loss_v3.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 1 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== Experiment v3 (2/2): bbox-split eval on v3 checkpoints (250-3000) ==="
& "$PSScriptRoot\eval_bbox_split_nyuv2_v3.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 2 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== Experiment v3 done ==="
exit 0
