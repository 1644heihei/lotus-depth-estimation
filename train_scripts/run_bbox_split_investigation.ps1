$ErrorActionPreference = "Stop"

Write-Host "=== Step 1/3: bbox-split eval on surviving v1 checkpoints (3500-8000) ==="
& "$PSScriptRoot\eval_bbox_split_nyuv2_v1.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 1 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== Step 2/3: fine-grained retrain (0-3000 steps, checkpoint every 250) ==="
& "$PSScriptRoot\train_lotus_d_object_bbox_loss_v2.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 2 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== Step 3/3: bbox-split eval on v2 checkpoints (250-3000) ==="
& "$PSScriptRoot\eval_bbox_split_nyuv2_v2.ps1"
if ($LASTEXITCODE -ne 0) { Write-Host "Step 3 failed ($LASTEXITCODE)"; exit $LASTEXITCODE }

Write-Host "=== All done ==="
exit 0
