set -u
# The Hypersim-trained selector was tested on NYUv2 and ScanNet but never on Hypersim.
# Without that row there is no way to tell whether it learned anything at all, so the
# -2.23% and +0.39% it scores elsewhere cannot be read as a transfer failure rather
# than a training failure. Same retention and same repair settings as every other run.
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
cd /d/lotus/lotus-depth-estimation
python apply_contour_selector.py --report_precision --retention 0.06 \
  --checkpoint output/selector_hs10k/best.pt \
  --dataset hypersim --model marigold_v2 \
  --out_dir "D:/lotus/data/oracle_cache/hs10k_sel_hs" > "$L/hs_in_ap.log" 2>&1
echo "apply=$?"; grep -a "適合率\|残存" "$L/hs_in_ap.log" | tail -2
python eval_yolo_contour_sharpening.py --dataset hypersim --model marigold_v2 \
  --mask_cache_dir "D:/lotus/data/oracle_cache/hs10k_sel_hs" --masks_are_contours \
  --band_px 2 --fill_radius 3 \
  --output_dir output/eval_hs10k_on_hs > "$L/hs_in_ev.log" 2>&1
echo "eval=$?"
python -c "
import json
s=json.load(open('output/eval_hs10k_on_hs/summary.json')); v=s['variants']; b=v['baseline']['bf1']
for k in v: print(f'  {k:16} bf1 {v[k][\"bf1\"]:.5f}  raw {(v[k][\"bf1\"]/b-1)*100:+.2f}%')
"
echo HS_IN_DONE
