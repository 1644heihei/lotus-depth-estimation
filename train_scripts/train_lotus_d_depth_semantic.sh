export MODEL_NAME="stabilityai/stable-diffusion-2-base"

# training dataset
export TRAIN_DATA_DIR_HYPERSIM=$PATH_TO_HYPERSIM_DATA
export TRAIN_DATA_DIR_VKITTI=$PATH_TO_VKITTI_DATA
export RES_HYPERSIM=576
export RES_VKITTI=375
export P_HYPERSIM=0.9
export NORMTYPE="trunc_disparity"

# semantic masks (same basename as RGB image)
export SEM_MASK_DIR_HYPERSIM=$PATH_TO_HYPERSIM_SEM_MASKS
export SEM_MASK_DIR_VKITTI=$PATH_TO_VKITTI_SEM_MASKS
export SEM_STRENGTH=0.25
export SEM_DROPOUT=0.2

# training configs
export BATCH_SIZE=8
export CUDA=0
export GAS=1
export TOTAL_BSZ=$(($BATCH_SIZE * ${#CUDA} * $GAS))

# model configs
export TIMESTEP=999
export TASK_NAME="depth"

# eval
export BASE_TEST_DATA_DIR="datasets/eval/"
export VALIDATION_IMAGES="datasets/quick_validation/"
export VAL_STEP=500

# output dir
export OUTPUT_DIR="output/train-lotus-d-${TASK_NAME}-semantic-bsz${TOTAL_BSZ}/"

accelerate launch --config_file=accelerate_configs/$CUDA.yaml --mixed_precision="fp16" \
  --main_process_port="13324" \
  train_lotus_d.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM \
  --resolution_hypersim=$RES_HYPERSIM \
  --train_data_dir_vkitti=$TRAIN_DATA_DIR_VKITTI \
  --resolution_vkitti=$RES_VKITTI \
  --prob_hypersim=$P_HYPERSIM \
  --mix_dataset \
  --random_flip \
  --norm_type=$NORMTYPE \
  --dataloader_num_workers=0 \
  --train_batch_size=$BATCH_SIZE \
  --gradient_accumulation_steps=$GAS \
  --gradient_checkpointing \
  --max_grad_norm=1 \
  --seed=42 \
  --max_train_steps=20000 \
  --learning_rate=3e-05 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --task_name=$TASK_NAME \
  --timestep=$TIMESTEP \
  --validation_images=$VALIDATION_IMAGES \
  --validation_steps=$VAL_STEP \
  --checkpointing_steps=$VAL_STEP \
  --base_test_data_dir=$BASE_TEST_DATA_DIR \
  --output_dir=$OUTPUT_DIR \
  --semantic_mask_dir_hypersim=$SEM_MASK_DIR_HYPERSIM \
  --semantic_mask_dir_vkitti=$SEM_MASK_DIR_VKITTI \
  --semantic_mask_strength=$SEM_STRENGTH \
  --semantic_dropout_p=$SEM_DROPOUT \
  --resume_from_checkpoint="latest"
