$ErrorActionPreference = "Stop"

$WorkDir = "D:\lotus\lotus-depth-estimation"
$LogDir = "D:\lotus\data\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "predepth_$Stamp.log"
$ErrLog = Join-Path $LogDir "predepth_$Stamp.err"
$MetaLog = Join-Path $LogDir "predepth_${Stamp}_meta.log"

$Python = (Get-Command python -ErrorAction Stop).Source
Set-Location $WorkDir

$ArgList = @(
    "utils/build_detail_train_dataset.py",
    "--rgb_dir=D:/lotus/data/hypersim_processed/train",
    "--output_dir=D:/lotus/data/hypersim_yolo_detections/train",
    "--core_model=jingheya/lotus-depth-d-v2-0-disparity",
    "--steps=predepth",
    "--skip_existing",
    "--half_precision"
)

@(
    "[$(Get-Date -Format o)] launching detached pre-depth job",
    "python=$Python",
    "stdout=$OutLog",
    "stderr=$ErrLog",
    "args=$($ArgList -join ' ')"
) | Out-File -FilePath $MetaLog -Encoding utf8

Start-Process -FilePath $Python -ArgumentList $ArgList -WorkingDirectory $WorkDir `
    -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -WindowStyle Hidden

"[$(Get-Date -Format o)] Start-Process issued, pid pending in $OutLog" | Out-File -FilePath $MetaLog -Append -Encoding utf8
Write-Output "meta=$MetaLog"
Write-Output "stdout=$OutLog"
