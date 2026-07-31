# 実装計画: bbox サイズ条件の追加

更新日: 2026-07-24（**採用: 12ch direct / 実装済み・学習は未実行**）  
前提: Approach A、Run B は **13ch（class VAE）** として残す

---

## 0. 採用決定（会話反映）

当初案 A は「13ch の class VAE はそのまま + size を直結 → 15ch」だったが、

- class_map の VAE は必須ではなく流用都合だった
- class と size の対応をシャープに保ちたい

ため、次を **採用** した:

```text
12ch = RGB(4) + pre VAE(4) + valid(1) + class(1) + size_w(1) + size_h(1)
```

- class / size_w / size_h は **VAE なし**（latent へ nearest）
- **同一ループで共塗り**（score 優先の winner を共有）→ 種類と大きさが画素で対応
- フラグ: `--enable_object_condition --enable_bbox_size_condition`
- 学習スクリプト: `train_scripts/train_lotus_d_detail_12ch.ps1`
- 出力: `output/train-lotus-d-detail-12ch-bsz8/`
- 13ch パス（VAE class）は後方互換のためコード上に残す

---

## 1. 目的

いまの `class_map` は bbox 内に **クラス ID だけ** を塗っている。  
物体の **幅・高さ（相対サイズ）を明示チャンネル** として入れ、詳細モデルが「大きい物体 / 小さい物体」を直接条件に使えるようにする。

既存 13ch 重みは残す（別 `output_dir`）。**13ch のまま resume は不可**。

---

## 2. 採用仕様（12ch）

### ラスタライズ

`utils/object_size_condition.py`:

```text
rasterize_class_and_size_maps(detections, H, W)
  -> class_map, size_w, size_h
```

- \(w_\text{norm}=(x2-x1)/W\), \(h_\text{norm}=(y2-y1)/H\)
- 重複は score 優先（class と size で **同一 winner**）
- `[-1,1]` 正規化後に Dataset へ

### UNet

- `extra_in_channels=8`（4→12）
- `downsample_condition_map` で class/size を latent 解像度へ
- dropout 時は pre / valid / class / size を同時ゼロ

### 実装済みファイル

| ファイル | 内容 |
|--|--|
| `utils/object_size_condition.py` | 共塗り |
| `utils/pre_depth_fusion.py` | `downsample_condition_map` |
| `utils/detail_train_dataset.py` | size を batch に |
| `train_lotus_d.py` | `--enable_bbox_size_condition` |
| `pipeline.py` | extra=8 分岐 |
| `infer_object_refined_depth.py` | 12/13ch 両対応 |
| `train_scripts/train_lotus_d_detail_12ch.ps1` | 学習起動 |

### 未実施

- ~~12k step 学習の実行~~ → 完了（best mid-eval @6k）
- ~~条件あり NYUv2 評価の用意~~ → スクリプト追加済み
  - 成果物生成: `train_scripts/build_nyuv2_detail_artifacts.ps1`
  - 評価: `eval_detail_conditioned_nyuv2.py` / `train_scripts/eval_detail_conditioned_nyuv2.ps1`
- フル 654 枚の条件ありスコア確定（成果物生成後に実行）

---

## 3. 方式候補（履歴）

| 案 | 内容 | 結果 |
|--|--|--|
| A' | class+size 全部 direct → **12ch** | **採用・実装** |
| A | class VAE のまま + size direct → 15ch | 不採用 |
| B | size も VAE | 非推奨 |
| C | 面積 1ch のみ | 後退用 |
| D | class_map にサイズ押し込み | やらない |

---

## 4. 成功基準（案）

- 12ch 学習が完走する
- 条件あり推論で 13ch 比で悪化しない
- 大小差のある物体で対応破綻が目視で増えない
