# Object Attention 状況整理

更新: 2026-08-27  
ブランチ: `feature/object-depth-regressor`（最新: `687eca1`）

> [!WARNING]
> **本ドキュメントの数値は 2 種類の推論解像度が混在している。**
>
> - **validation（654枚, 学習ループ内）** → `train_lotus_d.py` が `pipe()` に
>   `processing_res` を渡さないため、パイプライン既定の **768**
> - **本番 eval（522枚）** → `eval_regressor_predepth_nyuv2.py` を ps1 が
>   `--processing_res=512` で呼んでいたため **512**
>
> 512 は公式 Lotus のデフォルト（768）ではなく、regressor の ROI 抽出解像度が
> 評価スクリプトに流用された設定ミス（[`phase0_findings.md`](phase0_findings.md) §3.3.1）。
> 同一プロトコルの公式 Lotus は **512 で 0.05543 / 768 で 0.05000**。
>
> よって「本番 eval」表の公式 Lotus 0.0554 も **512 での値**であり、
> 公式本来の性能は 0.0500。**fine-tune 各版との差は記載より大きい。**
> 評価スクリプトは 768 に修正済み。
>
> 併せて、本手法の方向性そのものが Phase 0 で否定されている
> （物体は誤差の集中点ではない）。[`lotus_improvement_plan.md`](lotus_improvement_plan.md) を参照。

---

## やっていること

Hypersim で fine-tune した Lotus-D に、YOLO 検出 + object depth regressor 由来の **object token** を cross-attention で注入する 4ch conditioning。

- spatial bias（bbox ガウス）+ 3層 encoder（hidden=512, zero-init）
- regressor 特徴量を cache 化して学習加速
- 詳細設計: [`object_attention_review.md`](object_attention_review.md)

---

## 学習 run 一覧

| run | 出力 dir | steps | regressor | validation ベスト |
|-----|----------|-------|-----------|-----------------|
| 8k（旧 encoder） | `output/train-lotus-d-object-attention-4ch-8k` | 8k | v4_mask | step-6500 |
| 12k（レビュー対応後） | `output/train-lotus-d-object-attention-4ch-12k` | 12k | v4_mask | step-4500: **0.0641** / δ1 0.947 |
| **v5 ROI 20k（最新）** | `output/train-lotus-d-object-attention-4ch-v5roi-20k` | 20k | **v5_roi** | step-4500: **0.0668** / δ1 0.940 |

共通設定: dropout 0.05, warmup 500, spatial bias ON, batch 8  
起動: `train_scripts/train_lotus_d_object_attention_4ch.ps1`

---

## 評価結果

### validation（NYU full test, 654枚）

学習ループ内 500 step ごと。8k 本番 eval（522枚）とは **母集団が異なる**。

**v5 ROI 20k — TOP 3**

| step | abs_rel | δ1 | checkpoint 残存 |
|------|---------|-----|-----------------|
| 4500 | 0.0668 | 0.940 | ❌ 削除済み |
| 16000 | 0.0674 | 0.943 | ✅ |
| 15500 | 0.0674 | 0.941 | ❌ |

- 最終 step-20000: abs_rel 0.0748 / δ1 0.927
- 残存 checkpoint: 11000, 16000〜20000（limit=10）

### 本番 eval（522枚, min_detections≥1）

| 条件 | abs_rel | δ1 | 備考 |
|------|---------|-----|------|
| 公式 Lotus | **0.0554** | **0.971** | baseline |
| 8k + Attn OFF | 0.0640 | 0.955 | `output/eval-object-attention-8k-6500/` |
| 8k + Attn ON | 0.0639 | 0.955 | Attn 効果 ≈ noop |
| 12k / v5 ROI | **未実施** | — | — |

---

## コード状態（687eca1 で push 済み）

train/infer 不一致修正:

- eval/infer/visualize で `rgb_np` 必須化（ROI regressor）
- cache metadata 検証（regressor_dir 一致）
- `enable_object_spatial_bias` を encoder config に保存
- eval に `attention_cached` モード追加
- テスト: `tests/test_object_attention_condition.py`（35件 PASS）

---

## 所見

1. fine-tune 全体で公式 Lotus より **+0.008 程度** 悪化（8k 本番）
2. Attention ON/OFF 差は **実質なし**（8k）
3. 12k → v5 ROI 20k でも validation ベストは **step 4500 付近** に集中（過学習で後半悪化）
4. v5 ROI regressor への変更で 12k より validation は **やや悪化**（0.0641 → 0.0668）
5. ベスト checkpoint が `checkpoints_total_limit` で消える問題あり

---

## 次にやること

1. **v5 ROI 本番 eval** — 残存 checkpoint-16000 で 522枚、Attention ON/OFF + `attention_cached`
2. **12k 本番 eval** — 未実施のまま（checkpoint-10500 推奨だが要確認）
3. **checkpoint 保存** — validation ベストの自動保存 or limit 増加
4. 結果詳細: [`object_attention_training_results.md`](object_attention_training_results.md)（8k/12k 中心、要更新）

---

## 主要パス

| パス | 内容 |
|------|------|
| `utils/object_attention_condition.py` | encoder + feature 構築 |
| `utils/object_spatial_attention.py` | spatial bias |
| `eval_regressor_predepth_nyuv2.py` | 本番評価 |
| `D:/lotus/data/object_attention_cache_v2_roi/` | train/eval cache |
| `output/object_depth_regressor_v5_roi` | 現行 regressor |
