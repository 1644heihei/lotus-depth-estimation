set -u
# Two things the second-round review asked for that need measurement.
#
# 1. The random controls at the selector's own budget, three draws each. The masks used
#    before kept 4.1% against the selector's 6.0%, and arguing that the mismatch
#    favoured the control is not the same as matching it.
# 2. Grid 16 on the three remaining seeds. The grid differences are 0.15-1.17 points
#    while the seed standard deviation is 0.90, so a single-seed comparison cannot
#    support "91% of the gain" either way.
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
B="D:/lotus/data/oracle_cache/sam_auto48_sel_raw"
cd /d/lotus/lotus-depth-estimation

echo "########## 1. random controls, 3 draws each at 6% ##########"
for mode in pixels blocks; do
  tag=$( [ "$mode" = pixels ] && echo px || echo blk )
  for sd in 1 2 3; do
    echo "=== random/$mode seed $sd ==="
    python build_random_selection.py --mode "$mode" --seed "$sd" \
      --dataset nyuv2 --model marigold_v2 --budget_cache_dir "$B" \
      --out_dir "D:/lotus/data/oracle_cache/rnd_${tag}_s${sd}" \
      > "$L/rnd_${tag}_s${sd}_build.log" 2>&1
    echo "  build=$?"; grep -a "残存率" "$L/rnd_${tag}_s${sd}_build.log" | tail -1
    python eval_yolo_contour_sharpening.py --dataset nyuv2 --model marigold_v2 \
      --mask_cache_dir "D:/lotus/data/oracle_cache/rnd_${tag}_s${sd}" \
      --masks_are_contours --band_px 2 --fill_radius 3 \
      --output_dir "output/eval_rnd_${tag}_s${sd}" \
      > "$L/rnd_${tag}_s${sd}_ev.log" 2>&1
    echo "  eval=$?"
  done
done

echo "########## 2. grid 16 on seeds 1-3 ##########"
for sd in 1 2 3; do
  echo "=== grid16 seed $sd ==="
  python apply_contour_selector.py --report_precision --retention 0.06 \
    --checkpoint "output/contour_selector_s${sd}/best.pt" --pred_space log \
    --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred/res640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/sam_auto16_nyu" \
    --out_dir "D:/lotus/data/oracle_cache/sel_grid16_s${sd}" \
    > "$L/g16_s${sd}_ap.log" 2>&1
  echo "  apply=$?"; grep -a "適合率\|残存率" "$L/g16_s${sd}_ap.log" | tail -2
  python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred --processing_res 640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/sel_grid16_s${sd}" \
    --masks_are_contours --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "output/eval_grid16_sel_s${sd}" > "$L/g16_s${sd}_ev.log" 2>&1
  echo "  eval=$?"
done
echo REV2_FOLLOWUP_DONE
