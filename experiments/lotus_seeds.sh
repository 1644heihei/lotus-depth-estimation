set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
cd /d/lotus/lotus-depth-estimation
for s in s1 s2 s3; do
  echo "=== seed $s ==="
  python apply_contour_selector.py --report_precision --retention 0.06 \
    --checkpoint "output/contour_selector_$s/best.pt" \
    --pred_space disparity --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir "D:/lotus/data/oracle_cache/lotus_pred/res768" \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48 \
    --out_dir "D:/lotus/data/oracle_cache/lotus_${s}" > "$L/lotus_${s}_ap.log" 2>&1
  echo "  apply=$?"
  python eval_yolo_contour_sharpening.py \
    --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir "D:/lotus/data/oracle_cache/lotus_pred" --processing_res 768 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/lotus_${s}" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space disparity \
    --output_dir "output/eval_lotus_${s}" > "$L/lotus_${s}_ev.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
