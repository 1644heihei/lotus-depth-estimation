set -u
# Re-select the label threshold t without touching the test split.
#
# label_t_sweep.sh compared t by BF1 on the 654-frame test split, which is the split the
# paper reports on and claims the method never sees. The training runs' own val precision
# cannot replace it: the positive rate falls from 6.07% at t=10 to 2.13% at t=25, so
# precision drops mechanically with t and lift rises with it - the two criteria disagree.
# BF1 is the one criterion that does not move with t (it always sweeps 5-25% itself), so
# this scores BF1 on the 141 frames of the 40 scenes train_contour_selector.py held out
# of the 795 training frames.
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
V="datasets/nyuv2_selector_val_scenes.txt"
cd /d/lotus/lotus-depth-estimation
for t in 10 15 20 25; do
  if [ "$t" = "10" ]; then CK="output/contour_selector/best.pt"; else CK="output/selector_t$t/best.pt"; fi
  echo "=== label t=${t}%  ($CK) ==="
  python apply_contour_selector.py --report_precision --retention 0.06 --t "$t" \
    --checkpoint "$CK" --dataset nyuv2 --model marigold_v2 --split train \
    --out_dir "D:/lotus/data/oracle_cache/sel_val_t$t" > "$L/val_t${t}_ap.log" 2>&1
  echo "  apply=$?"
  python eval_yolo_contour_sharpening.py --dataset nyuv2 --model marigold_v2 --split train \
    --scenes_file "$V" \
    --mask_cache_dir "D:/lotus/data/oracle_cache/sel_val_t$t" --masks_are_contours \
    --band_px 2 --fill_radius 3 \
    --output_dir "output/eval_val_t$t" > "$L/val_t${t}_ev.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
