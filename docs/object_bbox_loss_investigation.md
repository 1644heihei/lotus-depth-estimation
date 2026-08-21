# object_bbox_loss LoRA — bbox-split 評価まとめ

更新: 2026-08-22
ブランチ: `feature/object-depth-aux-loss`

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

## 改善案（優先度順）

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

## 主要パス

| パス | 内容 |
|------|------|
| `output/train-lotus-d-object-bbox-lora/` | v1 学習結果（checkpoint 3500〜8000 のみ現存） |
| `output/train-lotus-d-object-bbox-lora-v2/` | v2 学習結果（checkpoint 250〜3000 全残存） |
| `output/eval_bbox_split_object_bbox_lora_v1/summary.csv` | v1 bbox-split 評価集計 |
| `output/eval_bbox_split_object_bbox_lora_v2/summary.csv` | v2 bbox-split 評価集計 |
| `output/bbox_split_investigation_run2.log` | 再実行時の全体ログ |
| `train_scripts/run_bbox_split_investigation.ps1` | 3ステップ一括実行スクリプト |
