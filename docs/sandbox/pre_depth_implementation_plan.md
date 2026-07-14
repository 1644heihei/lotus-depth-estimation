# Pre-Depth 実装計画

作成日: 2026-07-07
対象プラン: `docs/pre_depth_improvement_plan.md`

## 現状コードとの対応(調査結果)

| プランの要素 | 既存コード | 状態 |
|---|---|---|
| グローバル+クロップの2段推定 | `infer_roi_fusion.py`(global 推定 → bbox 拡張 → crop 推定 → soft blend) | 骨格あり |
| 最小二乗スケール・シフト合わせ | `evaluation/util/alignment.py` の `align_depth_least_square()` | 評価用にあり。**推論側は未実装(急所)** |
| bbox 抽出・余白付与 | `utils/semantic_mask_utils.py` の `connected_component_boxes()`, `expand_box()` | あり(余白 12% → 20〜30% に変更要) |
| マスク生成 | `utils/generate_semantic_masks.py`(YOLOv8-seg) | あり |
| 領域別評価(global/ROI/boundary abs_rel) | `evaluation/roi_depth_metrics.py` | あり |
| boundary F-score | なし | **新規実装** |
| pre-depth の潜在空間注入 | なし(`LotusDPipeline.__call__` は semantic_mask の early/latent fusion のみ) | **新規実装** |
| UNet conv_in 拡張 | なし(Lotus-G 用の 8ch 化は `train_lotus_g.py` にあるが別用途) | **新規実装** |
| GT からの擬似 pre-depth 合成 | なし | **新規実装** |
| マルチスケール勾配損失 | なし | **新規実装** |

注意: プラン文書の「UNet conv_in 8ch→12ch」は Lotus-G の話。**Lotus-D の UNet 入力は 4ch** なので、
実際の拡張は 4ch → 8ch(pre-depth latent)+1ch(有効マスクを latent 解像度に縮小)= 9ch とする。

---

## フェーズ 1: 学習不要の整合合成ベースライン(プラン Step 1)

**目的**: pre-depth 方式の情報上限を早期に見積もる。学習なしで結論が出る。

### 1-1. スケール・シフト合わせの実装

- 新規: `utils/depth_alignment.py`
  - `fit_scale_shift(src, ref, valid_mask) -> (scale, shift)`
    クロップ深度 `src` をグローバル深度 `ref` に最小二乗フィット。
    `evaluation/util/alignment.py::align_depth_least_square` を流用・簡略化。
  - 外れ値対策として上下 5% をトリムしてからフィット(オプション)。
- `infer_roi_fusion.py` を改修:
  - crop 深度貼り込み前に `fit_scale_shift()` を適用(`--align_mode {none,lstsq}`、既定 `lstsq`。
    `none` で従来動作と比較可能に)
  - `--roi_expand_ratio` 既定を 0.12 → 0.25 に(プランの 2〜3 割余白)
  - 複数 bbox が重なる画素は上書きでなく重み付き平均に変更
  - 貼り込みは soft weight(既存 `make_soft_weight`)でブレンド

### 1-2. 評価データの準備

- NYUv2 テスト画像(実写、YOLO が有効)に対し `utils/generate_semantic_masks.py` でマスク生成
  - 出力: `data/nyu_sem_masks/`(gitignore 済み領域)
  - YOLOv8n で開始、検出品質が低ければ yolov8x-seg に切替
- Hypersim 検証サンプルでも同様(こちらは検出漏れが多い前提で参考扱い)

### 1-3. 評価の実行

- boundary F-score を `evaluation/roi_depth_metrics.py` に追加
  (pred/GT それぞれの深度エッジを Canny 等で抽出し、許容距離付き precision/recall/F1)
- 比較 3 条件を同一プロトコルで評価:
  1. グローバル推定のみ(公式 Lotus-D、ベースライン)
  2. 整合合成なしの貼り込み(`--align_mode none`、前回方式相当)
  3. 整合合成あり(`--align_mode lstsq`)
- 指標: 全体 abs_rel / ROI abs_rel / boundary abs_rel / boundary F-score
- 出力: `output/roi_fusion_eval/` に条件別 CSV + サマリー

### 1-4. 判定(Go/No-Go)

- **Go**: 条件 3 が条件 1 に対し ROI・boundary 指標で改善し、全体 abs_rel が悪化しない
  → フェーズ 2 へ。改善幅を「学習で超えるべき上限の目安」として記録
- **No-Go**: 改善しない場合、学習に進む前にマスク品質(SAM 置換)や
  クロップ解像度・余白のアブレーションで原因を切り分ける

---

## フェーズ 2: 学習込みパイプライン(プラン Step 2〜5)

### 2-1. pre-depth 注入機構(Step 2)

- 新規: `utils/pre_depth_fusion.py`
  - `expand_unet_conv_in(unet, extra_channels=5, zero_init=True)`
    conv_in を 4ch → 9ch(pre-depth latent 4ch + 有効マスク 1ch)に拡張、追加重みゼロ初期化
  - `encode_pre_depth(vae, pre_depth)` — 1ch → 3ch 複製 + [-1,1] 正規化 → VAE encode
    (Lotus の深度ターゲット処理 `train_lotus_d.py` と同一の正規化 `trunc_disparity` を使う)
- `pipeline.py` の `LotusDPipeline.__call__` に `pre_depth=`, `pre_depth_valid_mask=` 引数を追加
  - pre_depth なしのときはゼロ latent + ゼロマスクを渡す(ゼロ初期化と併せて公式と同一出力)
- **サニティチェック(必須)**: 拡張直後・pre_depth=None の状態で公式
  `jingheya/lotus-depth-d-v2-0-disparity` と出力が一致することをテストで確認
  - 新規: `tests/test_zero_init_equivalence.py`(数枚の画像で最大絶対誤差 < 1e-4)

### 2-2. 学習時の擬似 pre-depth 合成(Step 3)

- 新規: `utils/pre_depth_synth.py`
  - `synth_pre_depth(gt_depth, obj_mask, rng) -> (pre_depth, valid_mask)`
    - 物体領域の GT 深度を切り出し
    - ランダムなスケール・シフト摂動(推論時の整合ズレを模擬)
    - ガウスぼかし、部分欠損(ランダム穴あけ)、ノイズ強度のランダム化
  - 条件ドロップアウト: 確率 0.3 で pre_depth 全体をゼロ化
- マスクは既存の YOLO 生成マスク(`build_semantic_mask_batch` の経路)を流用。
  品質が問題なら Hypersim GT semantic(オラクル)を検討(プラン補足参照)
- `train_lotus_d.py` の学習ループに組み込み
  (既存の semantic early/latent fusion コードパスとはフラグで排他:
  `--pre_depth_fusion` を追加し、`--semantic_fusion_mode` とは独立)

### 2-3. 損失と学習設定(Step 4, 5)

- マルチスケール勾配マッチング損失を `train_lotus_d.py` に追加
  (`--grad_loss_weight`、MiDaS 系の実装を参考に 4 スケール)
- 学習の出発点: `--pretrained_model_name_or_path=jingheya/lotus-depth-d-v2-0-disparity`
  (SD1.5 からの学習はしない。前回 NYUv2 で abs_rel 3.556 と崩壊した経路を排除)
- fp16 → bf16(`--mixed_precision=bf16`)、可能なら EMA 有効化
- 学習スクリプト: `train_scripts/train_lotus_d_pre_depth.ps1` を新規作成

### 2-4. 評価

- 学習途中: `utils/midtrain_eval.py` を pre-depth 対応に拡張し、
  checkpoint ごとに「pre-depth あり/なし」両方の abs_rel を記録
  (なし側が公式から劣化していないか常時監視 = 破滅的忘却の検知)
- 最終: フェーズ 1 と同一プロトコル(NYUv2 + 領域別指標)で
  「公式 / フェーズ1合成 / 学習済み pre-depth モデル」の 3 者比較

### 2-5. 判定

- **成功**: 学習済みモデルがフェーズ 1 の合成ベースラインを ROI・boundary 指標で上回る
- フェーズ 1 の上限に届かない場合: 条件ドロップアウト率・ノイズ強度・
  勾配損失重みのアブレーション → それでも駄目ならマスク品質(SAM / GT semantic)へ

---

## 実施順序まとめ

1. `utils/depth_alignment.py` + `infer_roi_fusion.py` 改修(フェーズ 1-1)
2. NYUv2 マスク生成 + boundary F-score 実装(1-2, 1-3)
3. 3 条件評価 → Go/No-Go 判定(1-4)
4. conv_in 拡張 + pipeline 改修 + ゼロ初期化等価性テスト(2-1)
5. 擬似 pre-depth 合成 + 学習ループ組み込み(2-2)
6. 損失追加 + 公式重みから学習開始(2-3)
7. 中間監視付き学習 → 最終 3 者比較(2-4, 2-5)

フェーズ 1 は GPU 推論のみで完結し、コード変更も局所的なので先行して短期間で判定する。
フェーズ 2 は 2-1 のサニティチェックを通過してから学習に進む(前回の崩壊の再発防止)。
