set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
H="D:/lotus/data/hypersim_holdout_rgb/eval"
cd /d/lotus/lotus-depth-estimation
# Both directions. 2% tests "keep only the most confident", which is the intuition;
# 23% matches Hypersim's own positive rate, which is what 4.5 claimed the fix was
# and never tested on Hypersim itself.
for r in 0.02 0.06 0.15 0.23 0.40; do
  tag="${r/0./r}"
  echo "=== retention $r ==="
  python apply_contour_selector.py --report_precision --retention "$r" \
    --checkpoint output/contour_selector/best.pt --pred_space log \
    --dataset hypersim --rgb_dir "$H" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_hypersim/res1024 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_hypersim \
    --out_dir "D:/lotus/data/oracle_cache/hs_sweep_$tag" > "$L/hs_${tag}_ap.log" 2>&1
  echo "  apply=$?"; grep -a "適合率" "$L/hs_${tag}_ap.log" | tail -1
  python eval_yolo_contour_sharpening.py --dataset hypersim --rgb_dir "$H" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_hypersim --processing_res 1024 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/hs_sweep_$tag" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "output/eval_hs_sweep_$tag" > "$L/hs_${tag}_ev.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
