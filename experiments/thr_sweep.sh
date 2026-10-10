set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
cd /d/lotus/lotus-depth-estimation
# The control must be the BEST a hand-crafted threshold can do, so every candidate
# feature is measured on the full split rather than the one the AUC sweep liked.
for f in dgrad dgrad_max5 drange5 drange9; do
  echo "=== $f ==="
  python build_threshold_selection.py --report_precision --feature $f \
    --dataset nyuv2 --rgb_dir "$N" --pred_space log \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred/res640 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48 \
    --out_dir "D:/lotus/data/oracle_cache/thr_mv2_$f" > "$L/thr_${f}_build.log" 2>&1
  echo "  build=$?"
  python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred --processing_res 640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/thr_mv2_$f" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "output/eval_thr_mv2_$f" > "$L/thr_${f}_eval.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
