set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
S="D:/lotus/data/scannet_rgb_only/val"
R="D:/lotus/data/marigold_depth_eval/scannet"
cd /d/lotus/lotus-depth-estimation

echo "=== 1/3 ScanNet 393 枚（136 シーン）で学習 ==="
rm -rf output/selector_scannet
python train_contour_selector.py --dataset scannet --rgb_dir "$S" \
  --scannet_split datasets/scannet_indomain_train.txt --scannet_root "$R" \
  --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet/res640 \
  --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_scannet \
  --val_frac 0.16 --epochs 30 --batch_size 4 --retention 0.06 \
  --output_dir output/selector_scannet > "$L/sn_in_train.log" 2>&1
echo "  train=$?"; grep -a "frames\|UNet\|最良" "$L/sn_in_train.log" | head -4

echo "=== 2/3 held-out 407 枚（135 シーン）に適用 ==="
python apply_contour_selector.py --report_precision --retention 0.06 \
  --checkpoint output/selector_scannet/best.pt --pred_space log \
  --dataset scannet --rgb_dir "$S" \
  --scannet_split datasets/scannet_indomain_test.txt --scannet_root "$R" \
  --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet/res640 \
  --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_scannet \
  --out_dir D:/lotus/data/oracle_cache/sn_indomain > "$L/sn_in_ap.log" 2>&1
echo "  apply=$?"; grep -a "適合率\|残存率" "$L/sn_in_ap.log" | tail -2

echo "=== 3/3 採点（選別あり / なし、同じ 407 枚）==="
for tag in sn_indomain sam_auto48_scannet; do
  out="output/eval_snin_$( [ "$tag" = "sn_indomain" ] && echo sel || echo nosel )"
  python eval_yolo_contour_sharpening.py --dataset scannet --rgb_dir "$S" \
    --scannet_split datasets/scannet_indomain_test.txt --scannet_root "$R" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet --processing_res 640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/$tag" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "$out" > "$L/sn_in_ev_$(basename $out).log" 2>&1
  echo "  $tag -> $out exit=$?"
done
echo ALL DONE
