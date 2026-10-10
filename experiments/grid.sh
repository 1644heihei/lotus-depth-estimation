set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
cd /d/lotus/lotus-depth-estimation
for g in 16 32; do
  echo "=== SAM grid $g: 輪郭を作る ==="
  python build_sam_masks.py --dataset nyuv2 --rgb_dir "$N" --mode automatic \
    --points_per_crop $g --out_dir "D:/lotus/data/oracle_cache/sam_auto${g}_nyu" \
    > "$L/grid${g}_sam.log" 2>&1
  echo "  sam=$?"

  echo "=== SAM grid $g: 選別器を適用 ==="
  python apply_contour_selector.py --report_precision --retention 0.06 \
    --checkpoint output/contour_selector/best.pt --pred_space log \
    --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred/res640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/sam_auto${g}_nyu" \
    --out_dir "D:/lotus/data/oracle_cache/sel_grid${g}" > "$L/grid${g}_ap.log" 2>&1
  echo "  apply=$?"; grep -a "適合率\|残存率" "$L/grid${g}_ap.log" | tail -2

  echo "=== SAM grid $g: 採点（選別あり / なし）==="
  for tag in "sel_grid${g}" "sam_auto${g}_nyu"; do
    out="output/eval_grid${g}_$( [ "$tag" = "sel_grid${g}" ] && echo sel || echo nosel )"
    python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
      --pred_cache_dir D:/lotus/data/marigold_v2_pred --processing_res 640 \
      --mask_cache_dir "D:/lotus/data/oracle_cache/$tag" --masks_are_contours \
      --band_px 2 --fill_radius 3 --align_space log \
      --output_dir "$out" > "$L/grid${g}_ev_$(basename $out).log" 2>&1
    echo "  $tag -> $out exit=$?"
  done
done
echo ALL DONE
