# Detail 学習ラン記録（重みパス / 学習方法）

更新日: 2026-07-24

あとから見返す用。実験ディレクトリは上書きせず残す想定。

---

## 共通前提（方式 A）

- **コア**: 公式 Lotus-D `jingheya/lotus-depth-d-v2-0-disparity`（独自コア学習なし）
- **条件データ（オフライン）**: YOLO → pre-depth / valid_mask → class_map  
  - 生成: `utils/build_detail_train_dataset.py`  
  - Hypersim 出力: `D:/lotus/data/hypersim_yolo_detections/train/`
- **RGB 元データ**: `D:/lotus/data/hypersim_processed/train`
- **学習スクリプト**: `train_scripts/train_lotus_d_detail.ps1`  
  - 起動例: `powershell -File train_scripts/train_lotus_d_detail.ps1`  
  - 実体は `accelerate launch ... train_lotus_d.py ...`
- **注意**: 学習中の NYUv2 中間評価は当時 **条件なし（ゼロ埋め）**。スコアは「条件あり精緻化」ではなくモニタ用。
- **注意（2026-08-27 追記）**: 本ドキュメントの mid-eval 値は `train_lotus_d.py` 経由のため
  **`processing_res=768`**（パイプライン既定）。一方 `eval_regressor_predepth_nyuv2.py` を
  使った本番評価系のドキュメントは **512**（設定ミス）で、両者は直接比較できない。
  詳細は [`phase0_findings.md`](phase0_findings.md) §3.3.1。

---

## Run A — 9ch（pre-depth + valid、class_map なし）

### 入力チャンネル

`RGB latent(4) + pre-depth latent(4) + valid(1) = 9ch`  
追加 ch はゼロ初期化。

### 主な設定

| 項目 | 値 |
|--|--|
| steps | 20000 |
| batch | 8 |
| LR | 3e-5（constant） |
| precision | bf16 |
| pre_depth dropout | 0.3 |
| detection score thr（学習時） | 0.5 |
| grad_loss_weight | 0.1 |
| RGB 再構成 | OFF（`--disable_rgb_reconstruction`） |
| object condition | OFF |

### 重みパス

リポジトリ相対（実体は `D:\lotus\lotus-depth-estimation\` 配下）:

| 用途 | パス |
|--|--|
| **最終パイプライン** | `output/train-lotus-d-detail-bsz8/` |
| UNet 重み | `output/train-lotus-d-detail-bsz8/unet/diffusion_pytorch_model.safetensors` |
| 中間 ckpt（例） | `output/train-lotus-d-detail-bsz8/checkpoint-12000/` |
| 中間評価 | `output/train-lotus-d-detail-bsz8/evaluation-XXXXX/` |

参考（条件なし mid-eval）: best absrel **0.0705 @ 12k**、final **0.0833 @ 20k**

### 推論時の指定例

```text
--detail_model=output/train-lotus-d-detail-bsz8
```

---

## Run B — 13ch（+ class_map）

### 入力チャンネル

`RGB(4) + pre-depth(4) + valid(1) + class_map latent(4) = 13ch`  
`--enable_object_condition` で有効化。

### class_map の中身（この時点）

- bbox 矩形内に **クラス ID のみ** を塗る（`utils/object_condition.py`）
- **bbox の幅・高さの数値チャンネルは未投入**
- 学習時は detections を score≥0.5 で再フィルタして valid / class_map を再構築

### 主な設定

| 項目 | 値 |
|--|--|
| steps | 12000 |
| batch | 8 |
| LR | 3e-5（constant） |
| precision | bf16 |
| pre_depth dropout | 0.3（class_map も同時ゼロ） |
| detection score thr | 0.5 |
| grad_loss_weight | 0.1 |
| RGB 再構成 | OFF |
| object condition | ON |
| 初期化 | 公式 Lotus-D から 13ch へ拡張（追加 ch ゼロ初期化） |

### 重みパス

| 用途 | パス |
|--|--|
| **最終パイプライン** | `output/train-lotus-d-detail-13ch-bsz8/` |
| UNet 重み | `output/train-lotus-d-detail-13ch-bsz8/unet/diffusion_pytorch_model.safetensors` |
| 中間 ckpt（例: best 候補） | `output/train-lotus-d-detail-13ch-bsz8/checkpoint-5000/` |
| 最終 step ckpt | `output/train-lotus-d-detail-13ch-bsz8/checkpoint-12000/` |
| 中間評価 | `output/train-lotus-d-detail-13ch-bsz8/evaluation-XXXXX/` |

学習ログ例: `D:/lotus/data/logs/train_detail_13ch_20260723_153730.err`

### 再学習の起動（同じ設定）

`train_scripts/train_lotus_d_detail.ps1` が Run B 向け設定済み。  
要点フラグ:

```text
--enable_pre_depth_fusion
--enable_object_condition
--pre_depth_dropout_p=0.3
--detection_score_thr=0.5
--grad_loss_weight=0.1
--disable_rgb_reconstruction
--output_dir=output/train-lotus-d-detail-13ch-bsz8/
```

**別実験をするときは `OUTPUT_DIR` を変える**（上書き防止）。

### 条件あり推論

```text
python infer_object_refined_depth.py
  --core_model=jingheya/lotus-depth-d-v2-0-disparity
  --detail_model=output/train-lotus-d-detail-13ch-bsz8
  --input_dir=...
  --output_dir=...
  --yolo_score_thr=0.5 --half_precision --disparity
```

（13ch 時は YOLO → pre-depth + class_map を渡す。スクリプト側で class_map 対応済み。）

---

## Run C — 12ch（class + size_w/h、物体マップは VAE なし）

実装日: 2026-07-24（学習は別途）

### 入力チャンネル

```text
RGB(4) + pre-depth VAE(4) + valid(1) + class(1) + size_w(1) + size_h(1) = 12ch
```

- class / size は **同一 score 優先で共塗り**（`rasterize_class_and_size_maps`）
- VAE は通さず latent へ nearest downsample（valid と同型）
- 既存 13ch とは非互換（別 `output_dir`）

### 主な設定

13ch とほぼ同じ（12k / bsz8 / dropout 0.3 / score 0.5 / RGB recon OFF）。追加フラグ:

```text
--enable_object_condition
--enable_bbox_size_condition
```

### パス・スクリプト

| 用途 | パス |
|--|--|
| 学習スクリプト | `train_scripts/train_lotus_d_detail_12ch.ps1` |
| 出力（予定） | `output/train-lotus-d-detail-12ch-bsz8/` |
| サイズ実装 | `utils/object_size_condition.py` |
| 計画 | `docs/sandbox/bbox_size_condition_plan.md` |

### 起動

```powershell
powershell -File train_scripts/train_lotus_d_detail_12ch.ps1
```

---

## 次の実験を足すとき

1. 新しい `output/train-lotus-d-detail-.../` を切る（既存 Run A/B/C は触らない）
2. このファイルに Run D としてパス・設定・日付を追記する
3. in_channels が変わる変更は **既存 ckpt のまま resume 不可**（新規 run）
