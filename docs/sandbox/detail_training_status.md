# Lotus 詳細深度（Approach A）— 現状共有メモ

更新日: 2026-07-24

**重みパスと学習方法の一覧（見返し用）:** [`detail_training_runs.md`](detail_training_runs.md)  
**bbox サイズ条件の実装計画:** [`bbox_size_condition_plan.md`](bbox_size_condition_plan.md)

## 1. 目的

単眼深度（Lotus-D）を、YOLO 由来の物体条件で **物体領域の深度を精緻化**する。  
学習は 2 段想定だが、今回は **公式 Lotus-D をコア代わりに使い、学習2（詳細モデル）のみ**実施。

---

## 2. 元の Lotus からの主な変更点

### モデル入力（UNet `conv_in`）

| | 公式 Lotus-D | 今回の詳細モデル（学習済み） |
|--|--|--|
| 入力 ch | **4ch**（RGB latent） | **9ch** |
| 内訳 | RGB(4) | RGB(4) + pre-depth(4) + valid mask(1) |
| 初期化 | — | 追加 5ch は **ゼロ初期化**（開始時は公式と同等出力を狙う） |

13ch（**実装・学習済み**）:

```
13ch = RGB(4) + pre-depth(4) + valid(1) + class_map latent(4)
```

- class_map は 1ch →（VAE用に 3ch 複製）→ latent 4ch
- class_map はクラス ID の塗りつぶしのみ（**bbox 幅・高さの明示チャンネルは未投入**）
- 詳細は [`detail_training_runs.md`](detail_training_runs.md)

### 学習データ

| | 公式 Lotus | 今回 |
|--|--|--|
| 教師 | RGB + GT 深度 | 同左 + **事前生成した条件** |
| 条件 | なし | pre-depth / valid_mask /（class_map は読んでいるが未使用） |
| YOLO | なし | **学習ループ外**（オフライン事前生成＝方式A） |

### 学習目的関数まわり（今回の実験設定）

- **RGB 再構成ブランチを OFF**（`--disable_rgb_reconstruction`）
  - 公式は depth + RGB 再構成の 2 本立て。今回は深度予測のみ（高速化のため）
- **勾配損失** `grad_loss_weight=0.1`
- **条件ドロップアウト** 30%（pre-depth を時々ゼロ）
- 検出スコア **≥0.5** を学習時にフィルタ（valid_mask / class_map 再構築）

### 追加した主なコード

- `utils/object_pre_depth.py` … global+crop → lstsq 整合 → pre-depth
- `utils/roi_fusion.py` / `utils/depth_alignment.py`
- `utils/pre_depth_fusion.py` … conv_in 拡張・VAE encode
- `utils/object_condition.py` … class_map 生成
- `utils/build_detail_train_dataset.py` … YOLO / pre-depth / class_map 一括生成
- `utils/detail_train_dataset.py` … 詳細学習用 DataLoader
- `train_scripts/train_lotus_d_detail.ps1`

---

## 3. データ準備の進捗（方式A）

対象: Hypersim train **59,543 枚**  
出力先: `D:/lotus/data/hypersim_yolo_detections/train/`

| 成果物 | 状態 |
|--|--|
| YOLO `*_detections.json` | 完了（生成時 conf≥0.25） |
| `*_pre_depth.npy` / `*_valid_mask.npy` | 完了（公式 Lotus-D で生成） |
| `*_class_map.npy` | 完了 |
| 学習時 score≥0.5 フィルタ | 実装済み（ファイルは作り直しせず読み込み時に適用） |

補足: crop RGB 自体は保存せず、融合後の pre-depth のみ保存。

---

## 4. 学習2の実験条件

- 初期重み: **公式** `jingheya/lotus-depth-d-v2-0-disparity`（学習1 checkpoint なし）
- データ: Hypersim + 上記 detail artifacts
- steps: 20,000 / batch 8 / bf16 / LR 3e-5
- RGB 再構成: **なし**
- 入力: **9ch**（class_map なし）
- 出力: `output/train-lotus-d-detail-bsz8/`

---

## 5. 結果（NYUv2、least-squares disparity 整合）

| モデル | abs_rel↓ | 備考 |
|--|--|--|
| 公式 Lotus-D | **≈0.051** | 過去評価（`eval_compare_nyuv2/official`） |
| フェーズ1 global（公式） | **0.050** | ROI融合なしが最良だった |
| 詳細モデル best | **0.0705** | **step 12000** |
| 詳細モデル final | **0.0833** | step 20000 |

要点:

- 全体 abs_rel は **公式より悪い**
- 曲線は単調改善ではなく、後半も 0.07〜0.12 で振動
- **step を増やしても公式 0.05 には届きにくそう**
- 現状の採用候補は final より **checkpoint-12000**

フェーズ1（学習なし ROI 融合）でも、NYUv2 では global 単体を超えられず No-Go 判定済み（`docs/sandbox/phase1_results.md`）。

---

## 6. いまの到達点 / 未着手

### できていること

- オフライン条件データ一式
- 9ch pre-depth 融合の学習パイプライン
- 20k step 学習完了 + 中間評価

### まだできていないこと

- NYUv2 条件データのオフライン生成 → 中間評価の条件あり化
- bbox サイズ等の明示チャンネル追加
- 学習1（独自コア）→ そのコアで pre-depth 再生成
- フル NYUv2 654 の条件あり本評価
- ROI / boundary 指標での詳細比較（mid-eval は条件なし全体 abs_rel 中心）
- RGB 再構成ありでの再実験

---

## 7. 相談したい論点（案）

1. **全体 abs_rel が公式より悪い主因は何か**
   - RGB 再構成削除 / Hypersim 過学習 / 条件の入れ方 / 評価が「条件なし推論」になっている点 など
2. **次は class_map（13ch）を足すべきか、先に 9ch の失敗分析か**
3. class_map は **VAE 4ch** でよいか、**1ch のまま**の方がよいか
4. 評価の主指標を全体 abs_rel から **ROI / boundary** に移すべきか（計画の成功基準に近い）
5. checkpoint-12000 をベースに続けるか、公式から 13ch でやり直しするか

---

## 8. 一言サマリ

> 9ch（pre-depth）→ 13ch（+class_map）まで学習済み。中間評価は条件なしのため「精緻化性能」は測れていない。条件ありスポット比較では公式より弱い。次は評価の条件あり化、または bbox サイズ等の条件強化。

---

## 参考パス

- **重み / 学習方法一覧**: `docs/sandbox/detail_training_runs.md`
- 計画書: `docs/object_condition_implementation_plan.html`
- フェーズ1結果: `docs/sandbox/phase1_results.md`
- 9ch 出力: `output/train-lotus-d-detail-bsz8/`
- 13ch 出力: `output/train-lotus-d-detail-13ch-bsz8/`
- オフラインデータ: `D:/lotus/data/hypersim_yolo_detections/train/`
- 学習スクリプト: `train_scripts/train_lotus_d_detail.ps1`
