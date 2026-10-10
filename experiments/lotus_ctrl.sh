set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
cd /d/lotus/lotus-depth-estimation

# The random selections are made over SAM contours, and SAM runs on RGB, so the same
# mask sets apply to any depth model. Only the scoring changes.
for tag in rand_px rand_blk; do
  echo "=== Lotus v1 x ${tag} ==="
  python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir D:/lotus/data/oracle_cache/lotus_pred --processing_res 768 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/sam_auto48_${tag}" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space disparity \
    --output_dir "output/eval_lotus_${tag}" > "$L/lotus_${tag}.log" 2>&1
  echo "  eval=$?"
done

echo "=== Lotus v1: 手作り閾値の選別を作る ==="
python build_threshold_selection.py --report_precision \
  --dataset nyuv2 --rgb_dir "$N" --pred_space disparity \
  --pred_cache_dir D:/lotus/data/oracle_cache/lotus_pred/res768 \
  --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48 \
  --out_dir D:/lotus/data/oracle_cache/thr_lotus > "$L/lotus_thr_build.log" 2>&1
echo "  build=$?"; tail -2 "$L/lotus_thr_build.log"

echo "=== Lotus v1 x 手作り閾値 ==="
python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
  --pred_cache_dir D:/lotus/data/oracle_cache/lotus_pred --processing_res 768 \
  --mask_cache_dir D:/lotus/data/oracle_cache/thr_lotus --masks_are_contours \
  --band_px 2 --fill_radius 3 --align_space disparity \
  --output_dir output/eval_lotus_thr > "$L/lotus_thr_eval.log" 2>&1
echo "  eval=$?"
echo ALL DONE
