# Semantic-aware Depth Estimation Report

## 1) Baseline summary
- Samples: 3
- ROI AbsRel: 1.1321690082550049
- ROI RMSE: 0.39705514907836914
- ROI Delta1: 0.2152253786722819

## 2) Class-wise failure tendencies
- mirror: roi_absrel=1.1305269002914429, boundary_absrel=1.1505873203277588
- glass: roi_absrel=1.135453224182129, boundary_absrel=1.0731712579727173

## 3) ROI-fusion ablation result
- Best run: `tmp_smoke\ablation\roi_fusion_w1`
- Mean ROI AbsRel improvement vs baseline: -0.04883877436319987
- ROI AbsRel (best): 1.1321690082550049
- ROI Delta1 (best): 0.2152253786722819

## 4) Recommended qualitative examples
- `im0` (class=mirror, roi_absrel_improvement=-0.007647514343261719)
- `im2` (class=mirror, roi_absrel_improvement=-0.06682324409484863)
- `im1` (class=glass, roi_absrel_improvement=-0.07204556465148926)

Use the following files for side-by-side figures:
- Baseline vis: `tmp_smoke\ablation\baseline\depth_vis`
- Fused vis: `tmp_smoke\ablation\roi_fusion_w1\depth_vis`
- ROI mask: `tmp_smoke\ablation\roi_fusion_w1\roi_mask`

## 5) Discussion prompts
- Which classes benefit most from ROI fusion?
- Are gains concentrated near object boundaries?
- How does semantic-mask-conditioned fine-tuning compare to inference-only fusion?
