$ErrorActionPreference = "Stop"
Set-Location "D:\lotus\lotus-depth-estimation"

$logDir = "D:\lotus\data\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "yolo_resume_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

Write-Output "[$(Get-Date -Format o)] YOLO resume started" | Tee-Object -FilePath $log

python utils/build_detail_train_dataset.py `
  --rgb_dir="D:/lotus/data/hypersim_processed/train" `
  --output_dir="D:/lotus/data/hypersim_yolo_detections/train" `
  --core_model="jingheya/lotus-depth-d-v2-0-disparity" `
  --steps="yolo" `
  --skip_existing 2>&1 | Tee-Object -FilePath $log -Append

Write-Output "[$(Get-Date -Format o)] YOLO resume finished" | Tee-Object -FilePath $log -Append
