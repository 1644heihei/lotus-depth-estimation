set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
cd /d/lotus/lotus-depth-estimation
echo "=== Marigold V2 / NYUv2 654 ==="
python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
  --pred_cache_dir D:/lotus/data/marigold_v2_pred --processing_res 640 \
  --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_sel_raw --masks_are_contours \
  --band_px 2 --fill_radius 3 --align_space log \
  --output_dir output/eval_see_mv2 > "$L/see_mv2.log" 2>&1
echo "  exit=$?"
echo "=== Lotus v1 / NYUv2 654 ==="
python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
  --pred_cache_dir D:/lotus/data/oracle_cache/lotus_pred --processing_res 768 \
  --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_sel_lotus_zs --masks_are_contours \
  --band_px 2 --fill_radius 3 --align_space disparity \
  --output_dir output/eval_see_lotus > "$L/see_lotus.log" 2>&1
echo "  exit=$?"
echo ALL DONE
