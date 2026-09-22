# 輪郭鋭化は Marigold V2 でも効くか — 実装計画

作成: 2026-09-22（初版）／ 同日 改訂: 前提を 7 点実測して確定
前提: [`contour_sharpening_findings.md`](contour_sharpening_findings.md) / [`whole_scene_contour_results.md`](whole_scene_contour_results.md)
測定は本リポジトリ、予測生成は `D:/lotus/marigold-v2`（fork: `1644heihei/marigold-v2`）

---

## 0. 何を決める測定か

Lotus v1 上で、輪郭を障壁にした深度の埋め直しが **正味 +18.4% BF1** を出している（学習ゼロ、abs_rel 不変）。
これが効くのは **Lotus v1 の境界が鈍いから**である。

Marigold V2 は Soft Edge Error を指標に据えて「毛・葉・髪のような細い縁」を狙う。
**新しい土台で境界が既に鋭いなら、+18.4% が働く余地は消える。**

この測定は「輪郭鋭化という方針に、まだ仕事が残っているか」を決める。
学習を伴わないので、9.3 日や 18.6 日を賭ける前に撃てる。

### 判定基準（着手前に固定）

| Marigold V2 の実測 | 意味 | 次の行動 |
|---|---|---|
| BF1 ≥ 0.12 かつ 鋭化の正味 ≤ +3% | 境界問題は土台交代で解決済み | **この方針を畳む。** 修論の軸を替える |
| 鋭化の正味 ≥ +10% | **結果が現行 SOTA でも成立** | 続行。「旧世代ではなく現行 SOTA を改善した」と書ける |
| 鋭化の正味が +3〜10% | 弱まったが生存 | オラクル上限を見て判断 |
| オラクル上限 ≥ +150% | 余地は v1 固有でなく構造的 | 学習側に投資する根拠になる |

---

## 1. 参照値（Lotus v1・NYUv2 654 枚・res768）

`output/eval_sharpen_auto48/summary.json`:

| | BF1 | abs_rel |
|---|---:|---:|
| baseline | 0.069336 | 0.049996 |
| 鋭化後 | 0.076486 | 0.050044 |
| 移設対照 | 0.063760 | 0.050033 |

生の増分 +10.31%、対照 −8.04%、**正味 +18.35%**。オラクル上限は同じ操作で **+279.9%**。

輪郭は `D:/lotus/data/oracle_cache/sam_auto48/`、`fill_radius=3`、`band_px=2`。
**radius は 3 以外にしない**（7 では +19.5% まで落ちる）。

---

## 2. 実測で確定した前提（改訂で追加）

初版は 7 点を未確認のまま書いていた。すべて潰した。

| # | 前提 | 実測結果 |
|---|---|---|
| 1 | 評価プロトコルが一致するか | **完全一致。** 両者 Eigen crop `[45:471, 41:601]`、深度 `1e-3`〜`10.0` |
| 2 | NYUv2 の実体 | 654 × 3 ファイル（`rgb_` / `depth_` / `filled_`）、全て **640×480**。16 の倍数なので丸めなしで推論可 |
| 3 | `sam_auto48` の被覆 | **654 件**（`<scene>/rgb_NNNN_seg.npz`）。過不足なし |
| 4 | 既存キャッシュの形式 | `res<N>/<scene>/rgb_NNNN_pred.npy`、**float16**、(480, 640)、値域 [0, 0.96] の**視差** |
| 5 | `align_to_gt` の空間 | **視差前提。** `pred_disp > 0` で絞るので log depth では半分落ちる |
| 6 | 鋭化コードへの影響範囲 | `base = align_to_gt(...)` の直後に鋭化。**aligner だけ差し替えれば鋭化は無改変** |
| 7 | `infer.py` の入力選択 | `image_dir.rglob("*")` で**全 `.png` を拾う**。フィルタ引数なし |

### 7 から生じる作業

`--image_dir` に NYUv2 を直接渡すと **1,962 枚**（`depth_*` と `filled_*` を含む）処理する。
3 倍の時間がかかり 2/3 は無意味。**654 枚の `rgb_*.png` だけを集めた staging ディレクトリを作る**
（シーン構造は保つ。約 330 MB、コピーで十分）。

### 6 の詳細 — 差し替えるのはここだけ

```python
# eval_object_oracle_ceiling.py:207  現行（視差）
def align_to_gt(pred_disp, gt, valid):
    gt_disp, gt_nn = depth2disparity(depth=gt, return_mask=True)
    valid_nn = valid & gt_nn & (pred_disp > 0)
    disp_aligned, _, _ = align_depth_least_square(gt_arr=gt_disp, pred_arr=pred_disp, ...)
    return np.clip(disparity2depth(np.clip(disp_aligned, 1e-3, None)), 1e-3, 10.0)
```

Marigold V2 に著者陣の変換がある（`marigold-v2/evaluation/src/util/alignment.py`）。
`align_depth_least_square` は**両リポジトリで同一関数**（どちらも Marigold V1 由来）。

```
depth2log_space(depth, eps=1e-6) -> log_space, non_negative_mask
log_space2depth(log_space, eps)  -> depth
```

これで `align_to_gt_log` を書き、`--align_space {disparity,log}` で切り替える。
**既定は `disparity`**（既存の全結果の再現性を壊さない）。

---

## 3. 段取り

### Phase 0 — 検証ゲート（通らなければ先に進まない）

1. staging ディレクトリを作る（654 枚の `rgb_*.png` のみ、シーン構造保持）
2. `scripts/infer.py --modality depth --image_dir <staging> --output_dir <out>`
   既定 `Log-stage2`、解像度はネイティブ 640×480
3. 出力 `predictions_npy/<scene>/rgb_NNNN.npy` を
   `D:/lotus/data/marigold_v2_pred/res480x640/<scene>/rgb_NNNN_pred.npy` へ **float16** で書く
4. `align_to_gt_log` を実装し、abs_rel と δ1 を測る

| ゲート | 値 |
|---|---|
| **通過** | abs_rel が **0.036 ± 0.004**（論文の NYUv2 AbsRel 3.6%） |
| 失敗 | aligner か出力仕様の解釈が誤り。**停止する** |

前提 1 で crop と深度範囲の一致を確認済みなので、**ずれるなら原因は aligner か
`Log-stage2` の出力仕様（符号の向き・log の底・eps）に限定される。**

2 回失敗したら `depth/Disparity-base` へ退避（アフィン不変**逆数深度**なので既存 aligner がそのまま使える）。
ただし Stage 1 recipe の 30k step で精度は本命より低く、論文の表と直接比較できない。

### Phase 1 — Marigold V2 の baseline BF1

同じ予測に `eval_boundary_f1.py` をかける（閾値 5→25 の 11 点、高 t 重み）。

**この数値はどの論文にも無い。** 論文が報告するのは AbsRel・δ1・SEE のみ。

### Phase 2 — 鋭化と移設対照

```
eval_yolo_contour_sharpening.py \
  --pred_cache_dir D:/lotus/data/marigold_v2_pred \
  --processing_res <staging に合わせた値> \
  --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48 \
  --masks_are_contours --fill_radius 3 --band_px 2 \
  --align_space log
```

**移設対照は必須。** v1 では生 +10.31% に対し対照が −8.04% だった。省けば過大評価になる。

### Phase 3 — オラクル上限

`eval_perfect_contour_ceiling.py` を同じ予測で走らせる。v1 では +279.9%。
**この数字が方針の生死を決める。**

---

## 4. 事故を繰り返さないための固定事項

| | 内容 | 過去の事故 |
|---|---|---|
| **フレーム数の明示検証** | 実行前に 654 個のキャッシュパスが存在することを assert する | `eval_yolo_contour_sharpening.py:149` は `if not pp.is_file(): continue` で**黙って飛ばす**。不足に気づかないまま baseline がずれる |
| **dtype** | float16 で保存する（既存キャッシュと同一） | 読み込みは `np.load(pp).astype(np.float64)` なので、float32 で保存すると fp16 往復が消えて v1 と非対称になる |
| **母集団** | NYUv2 654 枚すべて | 物体マスク 521 枚と全体 654 枚を混ぜ、baseline が 0.0726 と 0.0693 に割れた |
| **採点コードは 1 つ** | Marigold V2 側に eval を複製しない | — |
| **fill_radius** | 3 固定。変えるなら必ず掃引 | 7 を掃引せずに選び、結論 3 件を撤回した |

---

## 5. 見積もり

| Phase | 作業 | 時間 |
|---|---|---|
| 0 | staging ＋ 推論 654 枚 ＋ aligner ＋ ゲート | 2〜3 時間 |
| 1 | BF1 | 15 分 |
| 2 | 鋭化 ＋ 対照 | 30 分 |
| 3 | オラクル上限 | 30 分 |

**合計 4 時間。学習なし。**

推論のメモリは 1024×688 で 20.7 GB だったので、640×480 なら余裕がある。
速度は未測定で、654 枚の所要は Phase 0 で初めて分かる。

残る最大の不確実性は `Log-stage2` の出力仕様である。値域 [−1, 1] と
「アフィン不変 log depth」までは README で確認したが、**符号の向き**
（log depth は距離とともに増加、disparity は減少）を取り違えると
abs_rel が再現しない。ゲートがそれを捕まえる。

---

## 6. この測定がもたらすもの

| 結果 | 修論への影響 |
|---|---|
| 鋭化が Marigold V2 でも効く | 主張が「現行 SOTA の改善」になる。**最良** |
| 効かないが上限は高い | 手法は要るが器具が足りない。学習側に投資する根拠 |
| 効かず上限も低い | この方針は畳む。**ただし「土台の進化が境界問題を解いた」ことを測定で示したのは結果である** |

3 番目でも書けるものが残る。
