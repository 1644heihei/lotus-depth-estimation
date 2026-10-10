set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar"
cd /d/lotus/lotus-depth-estimation
# ScanNet in-domain reaches 37.1% where NYUv2 reaches 50.2%, but it also trained on
# 393 frames against 795. Cutting NYUv2 to 393 separates the ground truth's quality
# from the amount of data; two seeds, because one seed is what caused the last
# correction.
for sd in 42 1; do
  echo "=== NYUv2 393 枚 seed $sd ==="
  rm -rf "output/selector_nyu393_s$sd"
  python train_contour_selector.py --dataset nyuv2 --rgb_dir "$N/train" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_train/res640 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_train \
    --max_images 393 --val_frac 0.16 --epochs 30 --batch_size 4 --retention 0.06 \
    --seed $sd --output_dir "output/selector_nyu393_s$sd" > "$L/nyu393_s${sd}_train.log" 2>&1
  echo "  train=$?"; grep -a "frames" "$L/nyu393_s${sd}_train.log" | head -1
  python apply_contour_selector.py --report_precision --retention 0.06 \
    --checkpoint "output/selector_nyu393_s$sd/best.pt" --pred_space log \
    --dataset nyuv2 --rgb_dir "$N/test" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred/res640 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48 \
    --out_dir "D:/lotus/data/oracle_cache/nyu393_s$sd" > "$L/nyu393_s${sd}_ap.log" 2>&1
  echo "  apply=$?"; grep -a "適合率" "$L/nyu393_s${sd}_ap.log" | tail -1
  python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N/test" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred --processing_res 640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/nyu393_s$sd" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "output/eval_nyu393_s$sd" > "$L/nyu393_s${sd}_ev.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
