$ErrorActionPreference = "Stop"

# Build Approach-A offline artifacts for NYUv2 test (654 images):
# YOLO detections + pre-depth/valid_mask + class_map

$WorkDir = "D:\lotus\lotus-depth-estimation"
$LogDir = "D:\lotus\data\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path "D:\lotus\data\nyuv2_detail_artifacts\test" | Out-Null

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
Remove-Item Env:HUGGING_FACE_HUB_TOKEN -ErrorAction SilentlyContinue

$RGB_DIR = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
$OUTPUT_DIR = "D:/lotus/data/nyuv2_detail_artifacts/test"
$CORE_MODEL = "jingheya/lotus-depth-d-v2-0-disparity"

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutLog = Join-Path $LogDir "nyuv2_detail_$Stamp.log"
$ErrLog = Join-Path $LogDir "nyuv2_detail_$Stamp.err"
$MetaLog = Join-Path $LogDir "nyuv2_detail_${Stamp}_meta.log"

$Python = (Get-Command python -ErrorAction Stop).Source
Set-Location $WorkDir

$ArgList = @(
    "utils/build_detail_train_dataset.py",
    "--rgb_dir=$RGB_DIR",
    "--output_dir=$OUTPUT_DIR",
    "--core_model=$CORE_MODEL",
    "--pattern=rgb_*.png",
    "--steps=all",
    "--skip_existing",
    "--half_precision",
    "--yolo_score_thr=0.25"
)

@(
    "[$(Get-Date -Format o)] launching detached NYUv2 detail artifact build",
    "python=$Python",
    "stdout=$OutLog",
    "stderr=$ErrLog",
    "args=$($ArgList -join ' ')"
) | Out-File -FilePath $MetaLog -Encoding utf8

Start-Process -FilePath $Python -ArgumentList $ArgList -WorkingDirectory $WorkDir `
    -RedirectStandardOutput $OutLog -RedirectStandardError $ErrLog -WindowStyle Hidden

"[$(Get-Date -Format o)] Start-Process issued" | Out-File -FilePath $MetaLog -Append -Encoding utf8
Write-Output "meta=$MetaLog"
Write-Output "stdout=$OutLog"
Write-Output "stderr=$ErrLog"
