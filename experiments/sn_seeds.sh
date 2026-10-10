set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
S="D:/lotus/data/scannet_rgb_only/val"
R="D:/lotus/data/marigold_depth_eval/scannet"
cd /d/lotus/lotus-depth-estimation
# The seed moves the train/val scene split as well as the init, so this captures
# split variance too - the same thing the four NYUv2 seeds captured.
for sd in 1 2 3; do
  echo "=== seed $sd ==="
  rm -rf "output/selector_sn_s$sd"
  python train_contour_selector.py --dataset scannet --rgb_dir "$S" \
    --scannet_split datasets/scannet_indomain_train.txt --scannet_root "$R" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet/res640 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_scannet \
    --val_frac 0.16 --epochs 30 --batch_size 4 --retention 0.06 --seed $sd \
    --output_dir "output/selector_sn_s$sd" > "$L/sn_s${sd}_train.log" 2>&1
  echo "  train=$?"
  python apply_contour_selector.py --report_precision --retention 0.06 \
    --checkpoint "output/selector_sn_s$sd/best.pt" --pred_space log \
    --dataset scannet --rgb_dir "$S" \
    --scannet_split datasets/scannet_indomain_test.txt --scannet_root "$R" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet/res640 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_scannet \
    --out_dir "D:/lotus/data/oracle_cache/sn_in_s$sd" > "$L/sn_s${sd}_ap.log" 2>&1
  echo "  apply=$?"; grep -a "適合率" "$L/sn_s${sd}_ap.log" | tail -1
  python eval_yolo_contour_sharpening.py --dataset scannet --rgb_dir "$S" \
    --scannet_split datasets/scannet_indomain_test.txt --scannet_root "$R" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet --processing_res 640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/sn_in_s$sd" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "output/eval_snin_s$sd" > "$L/sn_s${sd}_ev.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
