# 実装計画: bbox サイズ条件の追加

更新日: 2026-07-24  
前提: Approach A（オフライン YOLO + pre-depth + class_map）、現行 Run B は **13ch**

---

## 1. 目的

いまの `class_map` は bbox 内に **クラス ID だけ** を塗っている。  
物体の **幅・高さ（相対サイズ）を明示チャンネル** として入れ、詳細モデルが「大きい物体 / 小さい物体」を直接条件に使えるようにする。

既存 13ch 重みは残す（別 `output_dir` で新規学習）。in_channels が増えるため **13ch のまま resume は不可**。

---

## 2. 現状（比較用）

| 条件 | 表現 | UNet への入り方 | ch 寄与 |
|--|--|--|--|
| RGB | latent | そのまま | 4 |
| pre-depth | 連続マップ | VAE encode | 4 |
| valid | 0/1 マスク | latent 解像度へ downsample | 1 |
| class_map | クラス ID 塗り | normalize → VAE encode | 4 |
| **合計** | | | **13** |

bbox サイズは「塗られた矩形の広さ」としてしか伝わっていない。

---

## 3. 方式候補

### 案 A（推奨）: size_w / size_h を latent に直接 concat（+2 → **15ch**）

各 detection の bbox について:

- \(w_\text{norm} = (x2-x1) / W\)
- \(h_\text{norm} = (y2-y1) / H\)

を矩形内に定数塗り（重複は class_map と同様 **score が高い方を優先**）。  
背景は 0。値域は `[0,1]` → 学習入力は `*2-1` で `[-1,1]`。

UNet へは **VAE を通さず**、`valid_mask` と同じく latent 解像度へ resize（nearest）して 2ch concat。

```
15ch = RGB(4) + pre(4) + valid(1) + class_map(4) + size_w(1) + size_h(1)
```

**採る理由**

- サイズは低周波の幾何量。VAE を通すと量子化・破壊されやすい
- valid と同パターンで実装が小さい
- ch 増分が +2 のみ（VAE 経由だと +4〜8 になりがち）
- 既存 detections.json から **学習時オンザフライ生成可能**（オフライン再生成は任意）

### 案 B: size も VAE 経由（+4 → **17ch**）

`size` を 3ch に複製して class_map と同じく `encode_pre_depth_latents`。  
実装は均一だが、連続量に VAE は過剰で ch も増える → **非推奨**。

### 案 C: 面積 1ch のみ（+1 → **14ch**）

\(\sqrt{w_\text{norm} \cdot h_\text{norm}}\) や \(w\cdot h\) だけ。  
実装は最軽量だが、縦長/横長が区別できない → まずは案 A を推奨。後から削るのは容易。

### 案 D: class_map にサイズをエンコードして ch 据え置き

例: ID とサイズを 1ch に押し込む。意味が混ざる・resume できても学習意味が変わる → **やらない**。

---

## 4. 推奨仕様（案 A 詳細）

### 4.1 ラスタライズ

新規: `utils/object_size_condition.py`（または `object_condition.py` に追加）

```text
rasterize_size_maps(detections, H, W) -> (size_w: HxW float32, size_h: HxW float32)
```

- 塗りルール: class_map と同じ score 優先
- 空 detections → 全面 0
- オプション（初期は OFF）: `log1p` スケール `log1p(w_norm * max_side) / log1p(max_side)` など

### 4.2 正規化

```text
size_to_tensor(map01) = map01 * 2 - 1   # [-1, 1]
```

valid=0 の画素は size も 0（または dropout 時にまとめてゼロ）で揃える。

### 4.3 学習時フィルタ

既存どおり `detection_score_thr`（0.5）適用後の detections から  
`valid_mask` / `class_map` / **size_maps** を再構築。  
`pre_depth.npy` は変更しない。

### 4.4 UNet / pipeline

- `expand_unet_conv_in(..., extra_in_channels=11)`  
  （現行: 9ch 融合時 extra=5、13ch 時 extra=9 → **15ch 時 extra=11**）
- `train_lotus_d.py`: `size_w_values`, `size_h_values` を batch から読み、latent に cat
- `apply_pre_depth_dropout`: class_map と同様、dropout 時に size も同時ゼロ
- `pipeline.py`: `extra_channels in (5, 9, 11)` を分岐。`size_w` / `size_h` 引数追加（無しならゼロ）

### 4.5 オフライン成果物

**必須ではない**（detections.json があれば十分）。  
任意で `*_size_w.npy` / `*_size_h.npy` を `build_detail_train_dataset.py` に追加すると、デバッグと評価再現が楽。

初期実装は **オンザフライのみ** でよい。必要なら Phase 2 でキャッシュ。

---

## 5. 実装ステップ

### Step 0 — ドキュメント・出力先

- 本計画を確定
- 新 run の出力: `output/train-lotus-d-detail-15ch-bsz8/`（既存 9/13ch は触らない）
- `docs/sandbox/detail_training_runs.md` に Run C 追記欄を用意

### Step 1 — サイズマップ生成（単体）

1. `rasterize_size_maps` + 単体テスト（矩形塗り・overlap・正規化）
2. サンプル 1 枚で可視化（RGB | class_map | size_w | size_h）

完了条件: overlap と score 優先が class_map と一致すること。

### Step 2 — Dataset / Collate

1. `DetailTrainDataset.__getitem__` で size_w/h を生成・resize・normalize
2. collate に `size_w_values`, `size_h_values` を追加（shape `[B,1,H,W]`）
3. flag 無し時は出さない／ゼロでも可（後方互換）

### Step 3 — Train / Pipeline 配線

1. CLI: `--enable_bbox_size_condition`（`--enable_object_condition` 必須）
2. `extra = 11` に拡張、dropout 連動
3. `pipeline.py` / `infer_object_refined_depth.py` に size 渡し

完了条件: ダミー batch で forward が 15ch で通ること。

### Step 4 — 学習スクリプト

`train_scripts/train_lotus_d_detail.ps1` をコピー or 分岐:

- `OUTPUT_DIR=output/train-lotus-d-detail-15ch-bsz8/`
- `--enable_pre_depth_fusion --enable_object_condition --enable_bbox_size_condition`
- その他は 13ch と同じ（12k step, bsz8, dropout 0.3, score 0.5, RGB recon OFF）

初期化: **公式 Lotus-D から 15ch ゼロ拡張**（13ch からの部分コピーは任意・後回しで可）。

### Step 5 — 検証

1. スポット: 以前の 8 枚（公式 / 13ch / 15ch）
2. 可能なら NYUv2 条件あり評価（オフライン生成後）
3. `detail_training_runs.md` にパス・設定を追記

---

## 6. 触るファイル（予定）

| ファイル | 変更 |
|--|--|
| `utils/object_condition.py` または `utils/object_size_condition.py` | size ラスタライズ |
| `utils/detail_train_dataset.py` | size 読み込み・collate |
| `utils/pre_depth_fusion.py` | 必要なら downsample ヘルパ共用 |
| `train_lotus_d.py` | flag / expand / cat / dropout |
| `pipeline.py` | 15ch 分岐・引数 |
| `infer_object_refined_depth.py` | 推論時 size 生成・渡し |
| `train_scripts/train_lotus_d_detail.ps1`（または新 ps1） | 15ch 設定 |
| `docs/sandbox/detail_training_runs.md` | Run C 記録 |

任意:

| `utils/build_detail_train_dataset.py` | size npy キャッシュ |
| `docs/sandbox/detail_training_status.md` | ステータス更新 |

---

## 7. 学習・互換性メモ

| 項目 | 内容 |
|--|--|
| 既存 13ch 重み | **消えない**（別 dir）。ただし 15ch へはそのまま resume 不可 |
| オフライン Hypersim | detections 済みなら **再 YOLO 不要** |
| class_map npy | 変更不要（学習時再ラスタ） |
| 中間評価 | 現状どおり条件なしのままなら、size の効果は見えない → 条件あり評価が本命 |

---

## 8. リスクと緩和

| リスク | 緩和 |
|--|--|
| サイズが valid と冗長 | w/h はスカラー値、valid は 0/1。相関はあるが情報は異なる |
| 小さい bbox が効きにくい | 後で log スケールや min size clamp を試す |
| ch 増加で過学習・遅延 | +2 のみ。悪化時は案 C（1ch）に後退 |
| 評価が条件なしのまま | NYUv2 条件データ生成 + 条件あり eval を並行 |

---

## 9. 成功基準（案）

最低ライン（スポット or 部分セット）:

- 15ch 学習が最後まで落ちずに終わる
- 条件あり推論で **13ch 比で悪化しない**（absrel）
- 視覚的に、大小差のある物体境界で 13ch 以上の破綻がない

伸ばし目標:

- 条件あり NYUv2 で 13ch および公式に対し改善（まずは ROI / 物体領域）

---

## 10. やらないこと（この計画の範囲外）

- bbox 中心座標・向き・mask 輪郭の追加（サイズの次の候補）
- 独自コア（学習1）の再学習
- 13ch チェックポイントの削除・上書き
- class_map の意味書き換えによる ch 据え置きハック

---

## 11. 推奨スケジュール

1. Step 1–3（実装・ダミー forward）… 半日〜1 日  
2. Step 4 学習 12k … 既存と同様おおよそ半日〜1 日弱  
3. Step 5 比較・doc 追記 … 数時間  

**次のアクション**: 案 A（15ch, latent 直接 +2）でよいか確認 → OK なら Step 1 から実装。
