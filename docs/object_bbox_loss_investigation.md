# object_bbox_loss LoRA — bbox-split 評価まとめ

更新: 2026-08-27
ブランチ: `feature/object-depth-aux-loss`

> [!WARNING]
> **本ドキュメントの全数値は `processing_res=512` で取得されている。**
> 512 は公式 Lotus のデフォルト（**768**）ではなく、regressor の ROI 抽出解像度が
> 評価スクリプトに流用された設定ミスによるもの。詳細は
> [`phase0_findings.md`](phase0_findings.md) §3.3.1。
>
> - 同一プロトコルの公式 Lotus は **512 で 0.05543 / 768 で 0.05000**（−9.8%）
> - 実験群どうしの内部比較の公平性は保たれている（全て同じ 512）
> - **ただし公式 Lotus との比較としては、公式に不利なハンデが付いた条件だった**
> - 評価スクリプト（`train_scripts/eval_bbox_split_nyuv2_v*.ps1`）は 768 に修正済み。
>   再実行すると本ドキュメントの数値とは異なる（より良い）値になる。
>   本ドキュメントを再現するには各スクリプトの `$PROCESSING_RES = 512` に戻すこと。
>
> なお本調査の結論（bbox_loss の効果は最良 checkpoint で ON/OFF 差なし）は
> **512/768 の両方で検証済みの Oracle 測定によって独立に裏付けられている**
> （[`phase0_findings.md`](phase0_findings.md) §2）ため、この設定ミスの影響を受けない。

---

## 背景

`train_lotus_d.py` の `object_bbox_loss`(bbox内領域を重視した補助loss)付き LoRA 学習について、
NYUv2 test（654枚）を **bbox内(roi_abs_rel)** と **bbox外(bg_abs_rel)** に分けて評価し、
どの checkpoint が最良か、bbox_loss が意図通り ROI 精度を押し上げているかを調査した。

- 学習: `train_scripts/train_lotus_d_object_bbox_loss.ps1`（v1, 8000 step, checkpoint 500刻み）
- 学習: `train_scripts/train_lotus_d_object_bbox_loss_v2.ps1`（v2, 3000 step, checkpoint 250刻み。
  v1で削除された初期〜中盤 checkpoint を細かく再取得するための再学習）
- 評価: `train_scripts/eval_bbox_split_nyuv2_v1.ps1` / `eval_bbox_split_nyuv2_v2.ps1`
- 一括実行: `train_scripts/run_bbox_split_investigation.ps1`
- 評価スクリプト本体: `eval_regressor_predepth_nyuv2.py`（regressor: `output/object_depth_regressor_v5_roi`）
- 集計: `summarize_bbox_split_eval.py`

共通ハイパーパラメータ（v1 = v2）: `lora_rank=8`, `lora_alpha=16`, `object_bbox_loss_weight=1.0`,
`learning_rate=1e-4`, `lr_scheduler=constant`, `batch=8`, `resolution=512`, `grad_loss_weight=0.1`

---

## 実行時の問題と対応

初回実行（`run_bbox_split_investigation.ps1` を外側から `2>&1` でリダイレクトして起動）は
checkpoint-4000 の評価中（654枚中23枚目）に無エラーで停止した。原因は、PowerShell 5.1 で
native process（python）の stderr を `2>&1` でパイプすると各行が ErrorRecord に変換され、
スクリプト冒頭の `$ErrorActionPreference = "Stop"` と組み合わさって処理が停止したためと推定。

対応:
- `eval_bbox_split_nyuv2_v1.ps1` / `_v2.ps1` に **レジューム機能** を追加（`summary.json` が既に
  存在する checkpoint はスキップ）
- `train_lotus_d_object_bbox_loss_v2.ps1` に `--resume_from_checkpoint=latest` を追加
- 再実行時は `Start-Process` の `-RedirectStandardOutput` / `-RedirectStandardError`（OS レベルの
  リダイレクト）を使用し、PowerShell パイプラインの `2>&1` は使わない

再実行は正常に最後まで完走した（`output/bbox_split_investigation_run2.log`）。

---

## 結果

### v1: checkpoint 3500〜8000（500刻み）

| step | abs_rel | roi_abs_rel | bg_abs_rel | delta1 |
|------|---------|-------------|------------|--------|
| 3500 | 0.0729 | 0.0525 | 0.0689 | 0.9338 |
| 4000 | 0.0796 | 0.0568 | 0.0761 | 0.9194 |
| 4500 | 0.0790 | 0.0567 | 0.0750 | 0.9200 |
| 5000 | 0.0744 | 0.0538 | 0.0703 | 0.9287 |
| 5500 | 0.0750 | 0.0531 | 0.0715 | 0.9286 |
| 6000 | 0.0871 | 0.0626 | 0.0823 | 0.9049 |
| **6500** | **0.0713** | 0.0527 | **0.0678** | **0.9372** |
| 7000 | 0.0852 | 0.0612 | 0.0807 | 0.9074 |
| 7500 | 0.0732 | 0.0529 | 0.0701 | 0.9332 |
| 8000 | 0.0761 | 0.0555 | 0.0723 | 0.9266 |

v1範囲では横ばい〜微悪化を繰り返すのみで、明確な改善トレンドなし（6500が僅差で最良）。

### v2: checkpoint 250〜3000（250刻み、細粒度再学習）

| step | abs_rel | roi_abs_rel | bg_abs_rel | delta1 |
|------|---------|-------------|------------|--------|
| **250** | **0.0579** | **0.0457** | **0.0541** | **0.9669** |
| 500 | 0.0689 | 0.0487 | 0.0661 | 0.9480 |
| 750 | 0.0754 | 0.0532 | 0.0728 | 0.9305 |
| 1000 | 0.0664 | 0.0491 | 0.0636 | 0.9479 |
| 1250 | 0.0597 | 0.0465 | 0.0571 | 0.9606 |
| 1500 | 0.0690 | 0.0515 | 0.0665 | 0.9414 |
| 1750 | 0.0649 | 0.0483 | 0.0623 | 0.9487 |
| 2000 | 0.0643 | 0.0487 | 0.0619 | 0.9496 |
| 2250 | 0.0636 | 0.0481 | 0.0609 | 0.9523 |
| 2500 | 0.0652 | 0.0500 | 0.0626 | 0.9490 |
| 2750 | 0.0710 | 0.0513 | 0.0677 | 0.9376 |
| 3000 | 0.0737 | 0.0539 | 0.0702 | 0.9320 |

**checkpoint-250 が全checkpoint中で最良**（abs_rel=0.058, delta1=0.967）。v1最良の3500〜6500を
大きく上回る。step数が増えるほど（1250以降）悪化傾向が明確。

---

## 分析: bbox_loss は機能しているか

roi_abs_rel / bg_abs_rel の比を見ると、絶対値は両方とも学習が進むにつれ悪化する一方、
**比率は一貫して低下**しており、ROI が背景に対して相対的に改善し続けている:

| step | roi/bg 比 |
|------|-----------|
| 250 | 0.845 |
| 1250 | 0.814 |
| 2250 | 0.790 |
| 3000 | 0.768 |

→ **object_bbox_loss は狙い通り ROI 精度を相対的に押し上げている**が、その効果は
`learning_rate=1e-4` / `lr_scheduler=constant` による全体的な過学習（絶対値の悪化）に
飲み込まれてしまっている。

なお、`docs/object_attention_status.md` に記録された object attention 4ch 系の実験でも
同様に「validation ベストが学習ごく初期に集中し、後半で悪化」というパターンが見られており、
今回の object_bbox_loss LoRA と合わせて **本 fine-tune 手法全般に共通する過学習傾向**である
可能性が高い。

---

## 改善案（優先度順、v2時点）

1. **LRスケジュール変更**（最優先）— `constant` → `cosine` 等の decay ありスケジュールに変更、
   または LR 自体を `1e-4` → `3e-5` 程度に下げる。ピークが warmup 途中（250 step）にあることから、
   現状 LR が高すぎてベース重みから急速にドリフトしている可能性が高い。
2. **object_bbox_loss_weight を下げて再学習**（現状 1.0）— 全体劣化を抑えつつ ROI 改善効果だけを
   取り出せるか、0.3〜0.5 あたりでスイープして確認。
3. **250 step 未満をさらに細かく評価** — 現状 250 刻みのグリッドでは真の最適点を捉えられていない
   可能性がある。`eval_bbox_split_nyuv2_v2.ps1` の `$STEPS` を `50,100,150,200,250` 等に変更して
   再評価。
4. **ベースの fine-tune 自体の過学習** — `docs/object_attention_status.md` 記載の通り、
   公式 Lotus（abs_rel 0.0554）に対し fine-tune 後は総じて悪化している。Hypersim 単体への
   過適合が疑われるため、データ拡張強化・他データセット混合・ベース重みへの正則化（重み
   ドリフト制約）が有効な可能性がある。

---

## 追加実験: LRスイープ（v3〜v5）

改善案 #1 に沿って、v2（`constant`, LR=1e-4）を基準に scheduler と peak LR を段階的に変更。
他ハイパーパラメータ・checkpoint grid（250刻み, 250〜3000）はすべて共通。

- 学習: `train_scripts/train_lotus_d_object_bbox_loss_v3.ps1`（cosine, LR=1e-4）
- 学習: `train_scripts/train_lotus_d_object_bbox_loss_v4.ps1`（cosine, LR=3e-5）
- 学習: `train_scripts/train_lotus_d_object_bbox_loss_v5.ps1`（cosine, LR=1e-5）
- 評価: `eval_bbox_split_nyuv2_v3.ps1` / `_v4.ps1` / `_v5.ps1`（実行: `run_bbox_split_experiment_v3/v4/v5.ps1`）

### 比較（各実験の最良点 = step-250、および終端 step-3000 での劣化幅）

| 実験 | 設定 | step-250 abs_rel | step-250 delta1 | step-3000 abs_rel | 劣化率 |
|------|------|-------------------|-------------------|---------------------|--------|
| v2 | LR=1e-4, constant | 0.0579 | 0.9669 | 0.0737 | +27% |
| v3 | LR=1e-4, cosine | 0.0572 | 0.9674 | 0.0670 | +17% |
| v4 | LR=3e-5, cosine | 0.0556 | 0.9692 | 0.0667 | +20% |
| **v5** | **LR=1e-5, cosine** | **0.0554** | **0.9697** | **0.0584** | **+5%** |

**LR を下げるほど単調に改善**（v2→v3→v4→v5）。特に v5（LR=1e-5）は step-250〜3000 の劣化幅が
+5% まで大幅に縮小し、カーブが大きく平坦化した。ただし **どの設定でも最良 checkpoint は
一貫して step-250（= warmup 500 step の途中、ピーク LR に到達する前）のまま**であり、
「学習が進むほど良くなる」領域にはまだ到達していない。

---

## 追加実験: object_bbox_loss アブレーション（v6）— 重要な訂正

v2〜v5 は **すべて `object_bbox_loss_weight=1.0` で実行**しており、bbox loss を完全に切った
比較対照（コントロール）が存在しなかった。roi_abs_rel/bg_abs_rel 比の低下（上記「分析」節）を
「bbox_loss が機能している証拠」と解釈していたが、これはコントロールなしの誤った結論だった
可能性があるため、v4（LR=3e-5, cosine）と全く同じ設定・同じ seed=42 から
**`--object_bbox_detections_root` だけを外した**（＝bbox loss 項が一切追加されない、通常の
LoRA fine-tune）アブレーションを実行した。

- 学習: `train_scripts/train_lotus_d_object_bbox_loss_v6_ablation.ps1`
- 評価: `train_scripts/eval_bbox_split_nyuv2_v6_ablation.ps1`（実行: `run_bbox_split_experiment_v6_ablation.ps1`）

### v4（bbox_loss ON）vs v6（bbox_loss OFF、同一LR/schedule/seed）

| step | v4 abs_rel (ON) | v6 abs_rel (OFF) | v4 roi/bg比 | v6 roi/bg比 |
|------|-------------------|---------------------|--------------|--------------|
| **250（最良点）** | **0.05555** | **0.05574** | 0.860 | 0.864 |
| 1000 | 0.05898 | 0.06293 | 0.841 | 0.820 |
| 2000 | 0.06537 | 0.07117 | 0.780 | 0.768 |
| 3000 | 0.06673 | 0.06998 | 0.773 | 0.766 |

### 結論（v2〜v5の解釈の訂正）

1. **最良 checkpoint（step-250）では bbox_loss ON/OFF に実質的な差がない**（0.05555 vs
   0.05574、誤差レベル）。現状の運用で使うべき最良モデルにおいて、object_bbox_loss は
   精度に寄与していない。
2. **roi/bg 比が学習とともに低下する現象は、bbox_loss を切っても同様に起きる**（v6でも
   0.864→0.766、v4より僅かに下がり幅が大きい箇所もある）。したがって上記「分析」節の
   「object_bbox_loss が ROI を相対的に押し上げている」という解釈は誤りで、**この現象は
   LoRA fine-tuning 自体が持つ一般的な性質**（bbox_loss の有無と無関係）だったと訂正する。
3. bbox_loss に確認できた唯一の効果は、**step 数を増やしすぎた場合の劣化がやや緩やか**
   になる点（step-3000: v4 0.0667 vs v6 0.0700、約5%差）。ただし両者とも最良点（step-250）
   には及ばないため、実用上の意味は薄い。

→ **今回の実験レンジ（Hypersim fine-tune, LoRA rank=8, LR 1e-5〜1e-4）では、
object_bbox_loss を入れても入れなくても、最終的に採用すべき最良モデルの精度は変わらない。**
これまでの改善（v2→v5）はすべて「LR を下げて過学習を抑えた効果」であり、bbox_loss 自体の
寄与ではなかった。

---

## 今後の検討事項

1. **object_bbox_loss の設計見直し** — 現状の重み付け方式（bbox内 loss を単純に重み増し）
   では効果が確認できなかった。重みをさらに強くする（weight > 1.0）、bbox 境界も含める、
   あるいは pre_depth fusion（regressor 予測を conditioning channel として直接 UNet に
   入力する旧方式、`docs/object_attention_status.md` 参照）など、**loss だけでなく
   architecture/conditioning 側で物体情報を使う方式に戻す**方が有望な可能性がある。
2. **このアプローチ（loss reweighting のみ）を一旦保留し、上記 conditioning 系アプローチに
   注力する**選択肢もある。
3. 250 step 未満のさらに細かい評価（旧改善案 #3）は、bbox_loss 自体の効果が確認できなかった
   ため優先度を下げる。

---

## 主要パス

| パス | 内容 |
|------|------|
| `output/train-lotus-d-object-bbox-lora/` | v1 学習結果（checkpoint 3500〜8000 のみ現存） |
| `output/train-lotus-d-object-bbox-lora-v2/` | v2 学習結果（checkpoint 250〜3000 全残存） |
| `output/train-lotus-d-object-bbox-lora-v3/` | v3 学習結果（cosine, LR=1e-4） |
| `output/train-lotus-d-object-bbox-lora-v4/` | v4 学習結果（cosine, LR=3e-5） |
| `output/train-lotus-d-object-bbox-lora-v5/` | v5 学習結果（cosine, LR=1e-5） |
| `output/train-lotus-d-object-bbox-lora-v6-ablation/` | v6 学習結果（bbox_loss OFF アブレーション） |
| `output/eval_bbox_split_object_bbox_lora_v1/summary.csv` | v1 bbox-split 評価集計 |
| `output/eval_bbox_split_object_bbox_lora_v2/summary.csv` | v2 bbox-split 評価集計 |
| `output/eval_bbox_split_object_bbox_lora_v3/summary.csv` | v3 bbox-split 評価集計 |
| `output/eval_bbox_split_object_bbox_lora_v4/summary.csv` | v4 bbox-split 評価集計 |
| `output/eval_bbox_split_object_bbox_lora_v5/summary.csv` | v5 bbox-split 評価集計 |
| `output/eval_bbox_split_object_bbox_lora_v6_ablation/summary.csv` | v6 アブレーション評価集計 |
| `output/bbox_split_investigation_run2.log` | v1/v2 再実行時の全体ログ |
| `output/bbox_split_experiment_v3/v4/v5.log` | v3/v4/v5 実行ログ |
| `output/bbox_split_experiment_v6_ablation.log` | v6 アブレーション実行ログ |
| `train_scripts/run_bbox_split_investigation.ps1` | v1/v2 一括実行スクリプト |
| `train_scripts/run_bbox_split_experiment_v3/v4/v5.ps1` | v3/v4/v5 各実験の一括実行スクリプト |
| `train_scripts/run_bbox_split_experiment_v6_ablation.ps1` | v6 アブレーション一括実行スクリプト |
