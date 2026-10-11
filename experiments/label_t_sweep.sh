set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
cd /d/lotus/lotus-depth-estimation
# The label threshold was never compared against anything: t=10 was picked as the middle
# of BF1's 5-25 sweep. BF1 itself keeps sweeping whatever t the labels use, and 12.7% of
# its weight sits on steps below 10%, so a higher t should cost that weight. Measured
# rather than argued, at the retention the deployment uses.
for t in 15 20 25; do
  echo "=== label t=${t}% ==="
  rm -rf "output/selector_t$t"
  python train_contour_selector.py --dataset nyuv2 --model marigold_v2 --split train \
    --t "$t" --epochs 30 --batch_size 4 --retention 0.06 --val_scenes 40 \
    --output_dir "output/selector_t$t" > "$L/t${t}_train.log" 2>&1
  echo "  train=$?"; grep -a "最良の val" "$L/t${t}_train.log" | tail -1
  python apply_contour_selector.py --report_precision --retention 0.06 --t "$t" \
    --checkpoint "output/selector_t$t/best.pt" \
    --dataset nyuv2 --model marigold_v2 \
    --out_dir "D:/lotus/data/oracle_cache/sel_t$t" > "$L/t${t}_ap.log" 2>&1
  echo "  apply=$?"; grep -a "適合率" "$L/t${t}_ap.log" | tail -1
  python eval_yolo_contour_sharpening.py --dataset nyuv2 --model marigold_v2 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/sel_t$t" --masks_are_contours \
    --band_px 2 --fill_radius 3 \
    --output_dir "output/eval_t$t" > "$L/t${t}_ev.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
