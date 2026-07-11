# フェーズ1 結果: 物体 pre-depth パイプライン（学習不要）

実施日: 2026-07-10
実装: `utils/object_image.py`, `utils/semantic_mask_utils.py`, `utils/roi_fusion.py`,
`utils/depth_alignment.py`, `utils/boundary_metrics.py`, `infer_object_predepth.py`,
`utils/run_phase1_eval.py`

## 実験設定

- モデル: 公式 `jingheya/lotus-depth-d-v2-0-disparity`（fp16, 1-step regression）
- データ: NYUv2 テスト全 654 枚（Eigen crop 有効領域で評価）
- マスク: YOLOv8n-seg 事前生成（`data/nyu_sem_masks/`, 654 枚）
- 物体画像: **crop 方式**（bbox + 25% 余白、背景あり）→ Lotus-D で物体深度推定
- 整合: disparity 空間で最小二乗スケール・シフト（lstsq）
- 3 条件:
  - `global`: 全体推定のみ
  - `paste`: 物体深度を整合なしで貼り込み
  - `lstsq`: 物体深度を global に整合して pre-depth 化（貼り込み結果を最終出力）

## 結果（全 654 枚平均）

| 指標 | global | paste | lstsq |
|---|---|---|---|
| abs_rel(全体)↓ | **0.0500** | 0.0654 | 0.0525 |
| delta1 ↑ | **0.9711** | 0.9452 | 0.9688 |
| boundary F1 ↑ | **0.4478** | 0.4432 | 0.4418 |
| ROI abs_rel(物体領域)↓ | **0.0551** | 0.1437 | 0.0701 |
| 境界帯 abs_rel ↓ | **0.0716** | 0.1104 | 0.0772 |

検出あり: 621 / 654 枚

## 判定: **No-Go**（フェーズ2 学習の前提未達）

- `lstsq`（pre-depth 合成）は `paste` より大幅に改善するが、**全指標で `global` を上回れない**
- 物体領域でも global（0.055）< lstsq（0.070）
- 研究方針 Step 1〜2 の学習不要ベースラインでは、**公式 Lotus-D 1 回推定が最良**

## 出力

- 評価: `output/phase1_roi_fusion/summary.json`, `per_sample.csv`
- スモーク（5枚）: `output/phase1_roi_fusion_smoke/`

## 次の検討

1. `masked` 物体画像方式の ablation（`--object_image_mode masked`）
2. マスク品質改善（SAM / より大きい YOLO）
3. フェーズ2（潜在空間注入）に進む場合は、Step 1 の上限が低い点を認識した上で実施
