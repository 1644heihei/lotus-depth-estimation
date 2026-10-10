set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
S="D:/lotus/data/scannet_rgb_only/val"
cd /d/lotus/lotus-depth-estimation
for r in 0.02 0.04 0.07 0.12 0.20; do
  tag="${r/0./r}"
  echo "=== retention $r ==="
  python apply_contour_selector.py --report_precision --retention "$r" \
    --checkpoint output/contour_selector/best.pt \
    --dataset scannet --rgb_dir "$S" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet/res640 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_scannet \
    --out_dir "D:/lotus/data/oracle_cache/sn_sweep_$tag" > "$L/sweep_${tag}_ap.log" 2>&1
  echo "  apply=$?"
  python eval_yolo_contour_sharpening.py \
    --dataset scannet --rgb_dir "$S" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet --processing_res 640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/sn_sweep_$tag" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "output/eval_sn_sweep_$tag" > "$L/sweep_${tag}_ev.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
