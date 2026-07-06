# Lotus-D semantic depth training with YOLO masks (Windows)
$env:MODEL_NAME = "runwayml/stable-diffusion-v1-5"
$env:PATH_TO_HYPERSIM_DATA = "D:\lotus\data\hypersim_processed"
$env:PATH_TO_HYPERSIM_SEM_MASKS = "D:\lotus\data\hypersim_sem_masks"
$env:OUTPUT_DIR = "output/train-lotus-d-depth-semantic-mask-bsz8/"

$maskCount = (Get-ChildItem -Path $env:PATH_TO_HYPERSIM_SEM_MASKS -Recurse -Filter "*.png" -ErrorAction SilentlyContinue | Measure-Object).Count
$rgbCount = (Get-ChildItem -Path "$($env:PATH_TO_HYPERSIM_DATA)\train" -Recurse -Filter "rgb_cam_*.png" | Measure-Object).Count
Write-Host "[train] masks=$maskCount / rgb=$rgbCount"
if ($maskCount -lt $rgbCount) {
    Write-Error "Mask generation incomplete ($maskCount / $rgbCount). Wait for generate_semantic_masks.py to finish."
    exit 1
}

accelerate launch --mixed_precision="fp16" `
  --num_processes=1 `
  --num_machines=1 `
  --gpu_ids=0 `
  --main_process_port=13324 `
  train_lotus_d.py `
  --pretrained_model_name_or_path=$env:MODEL_NAME `
  --train_data_dir_hypersim=$env:PATH_TO_HYPERSIM_DATA `
  --resolution_hypersim=576 `
  --train_data_dir_vkitti="data/vkitti_processed" `
  --resolution_vkitti=375 `
  --prob_hypersim=1.0 `
  --random_flip `
  --norm_type="trunc_disparity" `
  --dataloader_num_workers=0 `
  --train_batch_size=4 `
  --gradient_accumulation_steps=2 `
  --gradient_checkpointing `
  --max_grad_norm=1 `
  --seed=42 `
  --max_train_steps=20000 `
  --learning_rate=3e-05 `
  --lr_scheduler="constant" `
  --lr_warmup_steps=0 `
  --task_name="depth" `
  --timestep=999 `
  --checkpointing_steps=500 `
  --output_dir=$env:OUTPUT_DIR `
  --semantic_mask_dir_hypersim=$env:PATH_TO_HYPERSIM_SEM_MASKS `
  --semantic_mask_dir_vkitti="data/vkitti_sem_masks" `
  --semantic_mask_strength=0.0 `
  --semantic_dropout_p=0.2 `
  --semantic_fusion_mode="early"
