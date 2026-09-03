# Object Attention Conditioning 学習・評価結果まとめ

更新日: 2026-08-19

本ドキュメントは、object attention conditioning（4ch Lotus-D + regressor 特徴量）の学習・評価結果を整理したものです。設計レビューは [`object_attention_review.md`](object_attention_review.md) を参照してください。

> [!WARNING]
> **本ドキュメントの数値は 2 種類の異なる推論解像度が混在している。**
>
> | 節 | 経路 | processing_res |
> |----|------|----------------|
> | §3 8k 本番評価（522枚） | `eval_regressor_predepth_nyuv2.py`（ps1 が `--processing_res=512` を明示） | **512** |
> | §4 12k 学習中 validation（654枚） | `train_lotus_d.py::run_evaluation` → `pipe()` に未指定 → パイプライン既定 | **768** |
>
> §3 の 512 は公式 Lotus のデフォルト（768）ではなく、regressor の ROI 抽出解像度が
> 評価スクリプトに流用された設定ミス（[`phase0_findings.md`](phase0_findings.md) §3.3.1）。
> 同一プロトコルの公式 Lotus は **512 で 0.05543 / 768 で 0.05000**（−9.8%）。
>
> したがって **§3 と §4 の数値は「評価母集団が違う」だけでなく「推論解像度も違う」**。
> 本文中の「母集団が異なるため直接比較しないこと」という注意書きは、
> **交絡要因が 2 つあった**と読み替えること。
>
> §3 の 8k 本番評価は公式 Lotus に不利な 512 で行われていたため、
> 「fine-tune が公式より +0.008 悪化」という差は**実際にはさらに大きい**。
> 評価スクリプトは 768 に修正済み。

---

## 1. 背景と実施した改善

[`object_attention_review.md`](object_attention_review.md) の指摘に基づき、以下を実装して再学習しました。

| 優先度 | 改善内容 | 実装 |
|--------|----------|------|
| 1 | Spatial attention bias | `utils/object_spatial_attention.py` — bbox 由来のガウス bias を attn2 cross-attention に加算（層ごとに動的生成） |
| 2 | Encoder 深化 | `ObjectAttentionEncoder`: 3層 MLP, hidden_dim=512, residual, out_proj ゼロ初期化 |
| 3 | Dropout 低減 | 0.2 → **0.05** |
| 4 | 学習 step 増加 | 8,000 → **12,000** |
| 5 | Warmup | 0 → **500 step** |

**旧 checkpoint（8k, 2層 encoder）との非互換**: 新 encoder アーキテクチャのため、8k 重みはそのままロード不可。

---

## 2. 学習設定比較

| 項目 | 8k 学習（旧） | 12k 学習（新） |
|------|---------------|----------------|
| 出力 dir | `output/train-lotus-d-object-attention-4ch-8k` | `output/train-lotus-d-object-attention-4ch-12k` |
| max_train_steps | 8,000 | **12,000** |
| object_attention_dropout_p | 0.2 | **0.05** |
| lr_warmup_steps | 0 | **500** |
| spatial bias | なし | **あり** |
| encoder | 2層, hidden=256 | **3層, hidden=512, zero-init out** |
| 学習時間 | 約 5 時間 | **約 6.6 時間** |
| ベースモデル | `jingheya/lotus-depth-d-v2-0-disparity` | 同左 |
| batch size | 8 | 8 |
| UNet LR / Encoder LR | 1e-5 / 1e-4 | 1e-5 / 1e-4 |
| 学習スクリプト | `train_scripts/train_lotus_d_object_attention_4ch.ps1` | 同左（12k 設定に更新済み） |

---

## 3. 8k 学習 — 本番評価（checkpoint-6500）

**評価条件**（`eval_regressor_predepth_nyuv2.py`）:

- データセット: NYUv2、検出 ≥1 の **522 枚**（654 枚中 132 枚除外）
- regressor: `output/object_depth_regressor_v4_mask`
- 4ch detail model（pre-depth 画像は未使用、object token のみ）
- 指標: least-square disparity alignment 後の abs_rel / δ1

| 条件 | abs_rel | δ1 | roi_abs_rel |
|------|---------|-----|-------------|
| **公式 Lotus**（unconditioned） | **0.0554** | **0.971** | 0.0447 |
| 8k fine-tune + Attention **OFF** | 0.0640 | 0.955 | 0.0495 |
| 8k fine-tune + Attention **ON** | 0.0639 | 0.955 | 0.0494 |

**所見（8k）**:

- fine-tune 全体で公式 Lotus より **+0.008 程度** 悪化
- Attention ON vs OFF の差は **実質 noop**（abs_rel 差 ≈ -0.0001、勝率 56% だが平均差なし）
- validation ベスト step: **6500**（8k 学習内）

**出力**: `output/eval-object-attention-8k-6500/`

---

## 4. 12k 学習 — 学習中 validation（NYU full test）

**評価条件**（学習ループ内、`train_lotus_d.py`）:

- データセット: **NYU full test**（`nyu_test_full`、654 枚）
- 500 step ごとに evaluation
- object attention cache 使用（検出 ≥1 フィルタあり）
- **注意**: 8k 本番評価（522 枚 subset）とは **評価母集団が異なる**ため、数値を直接比較しないこと

### 4.1 全 step の abs_rel / δ1

| step | abs_rel | δ1 |
|------|---------|-----|
| 500 | 0.1036 | 0.873 |
| 1000 | 0.1072 | 0.867 |
| 1500 | 0.0926 | 0.890 |
| 2000 | 0.0777 | 0.922 |
| 2500 | 0.1122 | 0.852 |
| 3000 | 0.0956 | 0.889 |
| 3500 | 0.0862 | 0.907 |
| 4000 | 0.0744 | 0.922 |
| **4500** | **0.0641** | **0.947** |
| 5000 | 0.0927 | 0.891 |
| 5500 | 0.0901 | 0.891 |
| 6000 | 0.0873 | 0.896 |
| 6500 | 0.1003 | 0.878 |
| 7000 | 0.0763 | 0.921 |
| 7500 | 0.0738 | 0.929 |
| 8000 | 0.0848 | 0.902 |
| 8500 | 0.1044 | 0.872 |
| 9000 | 0.0918 | 0.890 |
| 9500 | 0.0811 | 0.915 |
| 10000 | 0.1038 | 0.867 |
| 10500 | 0.0750 | 0.926 |
| 11000 | 0.0945 | 0.894 |
| 11500 | 0.0847 | 0.903 |
| 12000 | 0.0773 | 0.923 |

### 4.2 validation ベスト

| 順位 | step | abs_rel | δ1 |
|------|------|---------|-----|
| **1** | **4500** | **0.0641** | **0.947** |
| 2 | 7500 | 0.0738 | 0.929 |
| 3 | 4000 | 0.0744 | 0.922 |
| 4 | 10500 | 0.0750 | 0.926 |
| 5 | 7000 | 0.0763 | 0.921 |

**所見（12k validation）**:

- **step 4500 が abs_rel 最小**（0.0641）。8k 本番 eval（0.0639）と近い水準だが、評価母集団が異なる。
- 4500 以降は過学習気味で abs_rel が揺れて悪化する step が多い（6500, 8500, 10000 など）。
- spatial bias + 深い encoder により、**4500 時点で δ1=0.947** と 8k 全期間を上回る validation ピークを記録。

---

## 5. 保存 checkpoint

`checkpoints_total_limit=5` のため、**ディスク上に残っている checkpoint は以下のみ**:

| checkpoint | 備考 |
|------------|------|
| checkpoint-10000 | |
| checkpoint-10500 | validation 4位相当 |
| checkpoint-11000 | |
| checkpoint-11500 | |
| checkpoint-12000 | 最終 |

**ベスト step-4500 の重みは削除済み**。evaluation ログ（`evaluation-04500/`）と metrics ファイルは残存。

最終 pipeline 一式: `output/train-lotus-d-object-attention-4ch-12k/`（unet + object_condition_encoder）

---

## 6. 8k per-sample 分析（参考）

`output/eval-object-attention-8k-6500/per_sample_comparison.csv`

| 観点 | 結果 |
|------|------|
| Attention ON vs OFF | 勝率 56%、平均 abs_rel 差 **-0.0001**（実質 noop） |
| fine-tune vs 公式 Lotus | 約 65% の画像で公式に負け（+0.008 平均） |
| 失敗パターン | 公式は正常だが Attn 出力が全面赤（深度スケール破綻）— home_office, playroom 等 |
| 改善パターン | 公式が Hard（abs_rel>0.10）の画像で Attn がマシ |

可視化: `output/eval-object-attention-8k-6500/visualizations/best|worst/`

---

## 7. 実装上の修正（12k 学習開始時）

学習開始時に spatial bias 周りで以下を修正:

1. **bias テンソル代入形状** — `[B,1,Q,K]` への代入次元ミス
2. **UNet 層ごとの解像度差** — 固定解像度 bias が下位 block で不一致 → processor 内で **層ごとに動的生成** に変更
3. **pipeline 推論** — object token 連結後の embedding を num_text_tokens に使っていたバグ → text-only embedding を使用

関連ファイル:

- `utils/object_spatial_attention.py`
- `utils/object_attention_condition.py`
- `pipeline.py`
- `train_lotus_d.py`
- `eval_regressor_predepth_nyuv2.py`

---

## 8. 次のアクション

1. **12k 本番評価** — 残存 checkpoint（推奨: **10500** または **12000**）で `eval_regressor_predepth_nyuv2.py` を 522 枚 subset で実行し、8k・公式 Lotus と同一条件で比較
2. **Attention ON/OFF 比較** — 12k checkpoint で object token 有無の ablation
3. **checkpoint 保存ポリシー** — 次回学習では `checkpoints_total_limit` を増やすか、validation ベストを別途保存
4. **4500 相当の再現** — 必要なら step 4500 付近で短い追加 fine-tune、または best checkpoint 自動保存の導入

### 本番評価コマンド例

```powershell
# Attention ON（例: checkpoint-10500）
python eval_regressor_predepth_nyuv2.py `
  --detail_model output/train-lotus-d-object-attention-4ch-12k `
  --checkpoint_step 10500 `
  --regressor_dir output/object_depth_regressor_v4_mask `
  --condition_mode attention `
  --min_detections 1 `
  --output_dir output/eval-object-attention-12k-10500/attention-on
```

---

## 9. 関連パス一覧

| パス | 内容 |
|------|------|
| `docs/object_attention_review.md` | 設計レビュー原文 |
| `output/train-lotus-d-object-attention-4ch-8k/` | 8k 学習成果物 |
| `output/train-lotus-d-object-attention-4ch-12k/` | 12k 学習成果物 |
| `output/eval-object-attention-8k-6500/` | 8k 本番評価結果 |
| `train_scripts/train_lotus_d_object_attention_4ch.ps1` | 学習起動スクリプト |
| `eval_regressor_predepth_nyuv2.py` | 本番評価スクリプト |
| `scripts/analyze_per_sample_eval.py` | per-sample 分析 |
| `scripts/visualize_per_sample_cases.py` | 改善/失敗例可視化 |
