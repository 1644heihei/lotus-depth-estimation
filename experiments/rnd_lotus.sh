set -u
# The six budget-matched random masks, scored on Lotus as well, so both columns of the
# controls table come from the same draws at the same rate. The masks are built from SAM
# contours and the Marigold V2 selector's per-frame budget, which is the same budget the
# paper's Lotus column applies, so nothing about them is specific to the first model.
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
cd /d/lotus/lotus-depth-estimation
for tag in px blk; do
  for sd in 1 2 3; do
    echo "=== lotus x rnd_${tag}_s${sd} ==="
    python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
      --pred_cache_dir D:/lotus/data/oracle_cache/lotus_pred --processing_res 768 \
      --mask_cache_dir "D:/lotus/data/oracle_cache/rnd_${tag}_s${sd}" \
      --masks_are_contours --band_px 2 --fill_radius 3 --align_space disparity \
      --output_dir "output/eval_lotus_rnd_${tag}_s${sd}" \
      > "$L/lotus_rnd_${tag}_s${sd}.log" 2>&1
    echo "  eval=$?"
  done
done
echo RND_LOTUS_DONE
