set -u
L="C:/Users/nihei/AppData/Local/Temp/claude/d--lotus-lotus-depth-estimation/9deacae0-371a-4edc-a21a-0ac8833850a8/scratchpad"
N="C:/Users/nihei/lotus-depth-estimation/datasets/eval/depth/nyuv2/nyu_labeled_extracted.tar/test"
S="D:/lotus/data/scannet_rgb_only/val"
cd /d/lotus/lotus-depth-estimation

echo "=== 1 学習: ソース別 val、基準ごとに重みを保存 ==="
rm -rf output/selector_mix2
python train_contour_selector.py --mix_json datasets/mix_nyu_hs795.json \
  --crop 480 640 --val_frac 0.16 --epochs 30 --batch_size 4 --retention 0.06 \
  --output_dir output/selector_mix2 > "$L/mix2_train.log" 2>&1
echo "  train=$?"

for crit in macro nyuv2; do
  CK="output/selector_mix2/best_${crit}.pt"
  echo "=== ${crit}: NYUv2 ==="
  python apply_contour_selector.py --report_precision --retention 0.06 \
    --checkpoint "$CK" --pred_space log --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred/res640 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48 \
    --out_dir "D:/lotus/data/oracle_cache/mix2_${crit}_nyu" > "$L/mix2_${crit}_ap_nyu.log" 2>&1
  echo "  apply=$?"
  python eval_yolo_contour_sharpening.py --dataset nyuv2 --rgb_dir "$N" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred --processing_res 640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/mix2_${crit}_nyu" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "output/eval_mix2_${crit}_nyu" > "$L/mix2_${crit}_ev_nyu.log" 2>&1
  echo "  eval=$?"

  echo "=== ${crit}: ScanNet ==="
  python apply_contour_selector.py --report_precision --retention 0.06 \
    --checkpoint "$CK" --pred_space log --dataset scannet --rgb_dir "$S" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet/res640 \
    --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_scannet \
    --out_dir "D:/lotus/data/oracle_cache/mix2_${crit}_sn" > "$L/mix2_${crit}_ap_sn.log" 2>&1
  echo "  apply=$?"
  python eval_yolo_contour_sharpening.py --dataset scannet --rgb_dir "$S" \
    --pred_cache_dir D:/lotus/data/marigold_v2_pred_scannet --processing_res 640 \
    --mask_cache_dir "D:/lotus/data/oracle_cache/mix2_${crit}_sn" --masks_are_contours \
    --band_px 2 --fill_radius 3 --align_space log \
    --output_dir "output/eval_mix2_${crit}_scannet" > "$L/mix2_${crit}_ev_sn.log" 2>&1
  echo "  eval=$?"
done
echo ALL DONE
