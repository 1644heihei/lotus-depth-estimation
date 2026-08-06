$ErrorActionPreference = "Stop"
Set-Location "D:/lotus/lotus-depth-estimation"

$HYPER_RGB = "D:/lotus/data/hypersim_processed/train"
$HYPER_MANIFEST = "D:/lotus/data/hypersim_yolo_detections/train_manifest_score0.5.json"
$HYPER_DETECTIONS = "D:/lotus/data/hypersim_yolo_detections/train"
$HYPER_RECORDS = "D:/lotus/data/object_depth_records_v2/hypersim_train"
$NYU_RGB = "C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
$NYU_DETECTIONS = "D:/lotus/data/nyuv2_detail_artifacts/test"
$NYU_RECORDS = "D:/lotus/data/object_depth_records_v2/nyuv2_test"
$MODEL = "output/object_depth_regressor_v4_mask"

function Run-Python {
    param([string[]]$PythonArgs)
    & python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed with exit code $LASTEXITCODE`: python $($PythonArgs -join ' ')"
    }
}

# Re-run segmentation because the old JSON cache contains mask area but not its principal axis.
Run-Python @(
    "utils/build_detail_train_dataset.py",
    "--rgb_dir=$HYPER_RGB",
    "--output_dir=$HYPER_DETECTIONS",
    "--core_model=jingheya/lotus-depth-d-v2-0-disparity",
    "--steps=yolo",
    "--yolo_model=yolov8n-seg.pt",
    "--yolo_score_thr=0.5",
    "--yolo_device=0",
    "--overwrite"
)

Run-Python @(
    "utils/build_object_depth_records.py",
    "--rgb_dir=$HYPER_RGB",
    "--detections_root=$HYPER_DETECTIONS",
    "--output_root=$HYPER_RECORDS",
    "--manifest=$HYPER_MANIFEST",
    "--detection_score_thr=0.5",
    "--dataset=hypersim_train",
    "--depth_unit=mm",
    "--no_skip_existing"
)

Run-Python @(
    "utils/build_detail_train_dataset.py",
    "--rgb_dir=$NYU_RGB",
    "--output_dir=$NYU_DETECTIONS",
    "--core_model=jingheya/lotus-depth-d-v2-0-disparity",
    "--steps=yolo",
    "--pattern=rgb_*.png",
    "--yolo_model=yolov8n-seg.pt",
    "--yolo_score_thr=0.5",
    "--yolo_device=0",
    "--overwrite"
)

Run-Python @(
    "utils/build_object_depth_records.py",
    "--rgb_dir=$NYU_RGB",
    "--detections_root=$NYU_DETECTIONS",
    "--output_root=$NYU_RECORDS",
    "--pattern=rgb_*.png",
    "--detection_score_thr=0.5",
    "--dataset=nyuv2_test",
    "--depth_unit=mm",
    "--no_skip_existing"
)

Run-Python @(
    "train_object_depth_regressor.py",
    "--detail_root=$HYPER_RECORDS",
    "--manifest=$HYPER_MANIFEST",
    "--output_dir=$MODEL",
    "--feature_version=3",
    "--split_level=scene",
    "--max_depth_m=30",
    "--epochs=80",
    "--patience=12"
)

Run-Python @(
    "eval_object_depth_regressor.py",
    "--model_dir=$MODEL",
    "--split=nyuv2_test",
    "--detail_root=$NYU_RECORDS",
    "--rgb_dir=$NYU_RGB",
    "--pattern=rgb_*.png",
    "--output_dir=$MODEL/eval_nyuv2_test"
)
