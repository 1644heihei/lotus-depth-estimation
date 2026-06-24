export MODEL_NAME=${MODEL_NAME:-"runwayml/stable-diffusion-v1-5"}
export ACCELERATE_CONFIG=${ACCELERATE_CONFIG:-"accelerate_configs/01234567.yaml"}
export USE_LIBUV=0

# training dataset
export TRAIN_DATA_DIR_HYPERSIM=${PATH_TO_HYPERSIM_DATA:-"data/hypersim_processed"}
export TRAIN_DATA_DIR_VKITTI=${PATH_TO_VKITTI_DATA:-"data/vkitti_processed"}
export RES_HYPERSIM=576
export RES_VKITTI=375
export P_HYPERSIM=1.0
export NORMTYPE="trunc_disparity"

# semantic masks (same basename as RGB image)
export SEM_MASK_DIR_HYPERSIM=${PATH_TO_HYPERSIM_SEM_MASKS:-"data/hypersim_sem_masks"}
export SEM_MASK_DIR_VKITTI=${PATH_TO_VKITTI_SEM_MASKS:-"data/vkitti_sem_masks"}
export SEM_STRENGTH=0.0
export SEM_DROPOUT=0.2
export SEM_FUSION_MODE="early"

# optional: fetch only the required subset from HF before training
export HF_PREPARE_ON_DEMAND=${HF_PREPARE_ON_DEMAND:-0}
export HF_DATASET_REPO=${HF_DATASET_REPO:-"omrastogi/Hypersim-Processed"}
export HF_DATASET_SPLIT=${HF_DATASET_SPLIT:-"train"}
export HF_MAX_PAIRS=${HF_MAX_PAIRS:-0}
export HF_MAX_SCENES=${HF_MAX_SCENES:-0}
export HF_SCENE_PREFIX=${HF_SCENE_PREFIX:-""}
export HF_SLEEP_SEC=${HF_SLEEP_SEC:-0.0}
export HF_STREAMING_CACHE_MODE=${HF_STREAMING_CACHE_MODE:-0}

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

if [ "$HF_PREPARE_ON_DEMAND" = "1" ]; then
  echo "[train] Preparing subset from HF: repo=${HF_DATASET_REPO} split=${HF_DATASET_SPLIT} max_pairs=${HF_MAX_PAIRS}"
  python utils/prepare_hf_data.py \
    --repo_id="$HF_DATASET_REPO" \
    --split="$HF_DATASET_SPLIT" \
    --output_dir="$TRAIN_DATA_DIR_HYPERSIM" \
    --max_pairs="$HF_MAX_PAIRS" \
    --max_scenes="$HF_MAX_SCENES" \
    --scene_prefix="$HF_SCENE_PREFIX" \
    --sleep_sec="$HF_SLEEP_SEC"
fi

if [ "$HF_STREAMING_CACHE_MODE" = "1" ]; then
  TRAIN_DATA_DIR_HYPERSIM="hf://${HF_DATASET_REPO}/${HF_DATASET_SPLIT}?max_pairs=${HF_MAX_PAIRS}&scene_prefix=${HF_SCENE_PREFIX}"
  echo "[train] Using HF on-demand cache mode: ${TRAIN_DATA_DIR_HYPERSIM}"
fi

accelerate launch --mixed_precision="fp16" \
  --num_processes=1 \
  --num_machines=1 \
  --gpu_ids=$CUDA \
  --main_process_port="13324" \
  train_lotus_d.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir_hypersim=$TRAIN_DATA_DIR_HYPERSIM \
  --resolution_hypersim=$RES_HYPERSIM \
  --train_data_dir_vkitti=$TRAIN_DATA_DIR_VKITTI \
  --resolution_vkitti=$RES_VKITTI \
  --prob_hypersim=$P_HYPERSIM \
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
  --semantic_fusion_mode=$SEM_FUSION_MODE \
  --resume_from_checkpoint="latest"
