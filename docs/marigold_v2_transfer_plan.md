# 輪郭鋭化は Marigold V2 でも効くか — 実装計画

作成: 2026-09-22
前提: [`contour_sharpening_findings.md`](contour_sharpening_findings.md) / [`whole_scene_contour_results.md`](whole_scene_contour_results.md)
対象リポジトリ: 測定は本リポジトリ、予測生成は `D:/lotus/marigold-v2`

---

## 0. 何を決める測定か

Lotus v1 上で、輪郭を障壁にした深度の埋め直しが **正味 +18.4% BF1** を出している（学習ゼロ、abs_rel 不変）。
これが効くのは **Lotus v1 の境界が鈍いから**である。

Marigold V2 と Lotus-2 は、まさにその鈍さを潰しにきている。Marigold V2 は Soft Edge Error を
指標に据えて「毛・葉・髪のような細い縁」を狙う。**新しい土台で境界が既に鋭いなら、
+18.4% が働く余地は消える。**

この測定は「輪郭鋭化という方針に、まだ仕事が残っているか」を決める。
学習を一切伴わないので、9.3 日や 18.6 日を賭ける前に撃てる。

### 判定基準（着手前に固定する）

| Marigold V2 の実測 | 意味 | 次の行動 |
|---|---|---|
| BF1 ≥ 0.12 かつ 鋭化の正味 ≤ +3% | 境界問題は土台交代で解決済み | **この方針を畳む。** 修論の軸を替える |
| 鋭化の正味 ≥ +10% | **結果が現行 SOTA でも成立** | 続行。「旧世代ではなく現行 SOTA を改善した」と書ける |
| 鋭化の正味が +3〜10% | 弱まったが生存 | オラクル上限を見て判断 |
| オラクル上限 ≥ +150% | 余地は v1 固有でなく構造的 | 学習側に投資する根拠になる |

---

## 1. 参照値（Lotus v1・NYUv2 654 枚）

`output/eval_sharpen_auto48/summary.json` から:

| | BF1 | abs_rel |
|---|---:|---:|
| baseline | 0.069336 | 0.049996 |
| 鋭化後 | 0.076486 | 0.050044 |
| 移設対照 | 0.063760 | 0.050033 |

生の増分 +10.31%、対照 −8.04%、**正味 +18.35%**。
オラクル上限は同じ操作で **+279.9%**。

輪郭は `D:/lotus/data/oracle_cache/sam_auto48/`（SAM automatic、grid 48）。
パラメータは `fill_radius=3`、`band_px=2`。**radius は 3 以外にしない**
（7 では +19.5% まで落ちることを測定済み）。

---

## 2. 最大の技術的障害 — 表現空間が違う

`eval_object_oracle_ceiling.py:207` の `align_to_gt` は **視差前提**である。

```python
def align_to_gt(pred_disp, gt, valid):
    gt_disp, gt_nn = depth2disparity(depth=gt, return_mask=True)
    valid_nn = valid & gt_nn & (pred_disp > 0)        # ← 負値が落ちる
    disp_aligned, _, _ = align_depth_least_square(gt_arr=gt_disp, pred_arr=pred_disp, ...)
    return np.clip(disparity2depth(...), 1e-3, 10.0)
```

Lotus v1 は `norm_type=trunc_disparity` でアフィン不変**視差**を出すので整合する。
Marigold V2 の `Log-stage2` は アフィン不変 **log depth** を **[−1, 1]** で出す。
そのまま通すと `pred_disp > 0` で約半分の画素が落ち、しかも誤った空間で合わせる。

### 対処

Marigold V2 側に著者陣の変換関数がある（`marigold-v2/evaluation/src/util/alignment.py`）。

```
depth2log_space(depth, eps=1e-6)   -> log_space, non_negative_mask
log_space2depth(log_space, eps)    -> depth
align_depth_least_square(...)      ← Lotus 側と同一関数（どちらも Marigold V1 由来）
```

`align_to_gt` の視差変換を log 変換に差し替えた `align_to_gt_log` を書く。構造は同じ。

**重要**: 鋭化処理は `align_to_gt` の**後**、アフィン合わせ済みのメートル深度に対して走る
（`eval_yolo_contour_sharpening.py` は `base = align_to_gt(...)` してから `base` を鋭化する）。
したがって **aligner だけ差し替えれば、鋭化コードは一行も変えなくてよい。**

---

## 3. 段取り

### Phase 0 — 検証ゲート（ここが通らなければ以降は全部無意味）

**自作 aligner が論文の数値を再現するか確かめる。**

1. NYUv2 654 枚の RGB に `scripts/infer.py` を流す
   - `--modality depth`、既定の `Log-stage2`
   - 解像度は **640×480 のネイティブ**（どちらも 16 の倍数。1024×688 で 20.7 GB だったので余裕）
2. `align_to_gt_log` を実装
3. abs_rel と δ1 を測る

| ゲート | 値 |
|---|---|
| 通過 | abs_rel が **0.036 ± 0.004** に入る（論文の NYUv2 AbsRel 3.6%） |
| 失敗 | aligner か前処理が誤っている。**先に進まない** |

**留意**: 論文の 3.6% が Eigen crop を使っているかは未確認。crop が違えば数値がずれるので、
ずれた場合は crop の有無を切り替えて両方測り、どちらが一致するかで判定する。

### Phase 1 — Marigold V2 の baseline BF1

同じ予測に BF1 をかける（閾値 5→25 の 11 点、高 t 重み、`eval_boundary_f1.py` のまま）。

**この数値はどの論文にも無い。** 論文が報告するのは AbsRel・δ1・SEE のみ。

予測: Marigold V2 の境界が実際に鋭いなら、Lotus v1 の 0.0693 を大きく上回る。

### Phase 2 — 鋭化と移設対照

予測を `eval_yolo_contour_sharpening.py` に食わせる。

```
--pred_cache_dir  <Marigold V2 の予測を置いた場所>
--mask_cache_dir  D:/lotus/data/oracle_cache/sam_auto48
--masks_are_contours
--fill_radius 3  --band_px 2
```

必要な実装は 2 点のみ。

| | 内容 |
|---|---|
| キャッシュ配置 | `res<N>/<相対パス>_pred.npy` の形で書き出すアダプタ |
| aligner の切替 | `--align_space {disparity,log}` を追加し、既定は `disparity`（既存挙動を変えない） |

**移設対照は必須。** 輪郭を別の場所へずらして同じ操作をかけ、
「操作そのものが BF1 を上げる」分を差し引く。これを省くと過大評価になる
（v1 では生 +10.31% に対し対照が −8.04% だった）。

### Phase 3 — オラクル上限

`eval_perfect_contour_ceiling.py` を同じ予測で走らせ、GT 輪郭を使った上限を測る。

v1 では +279.9%。**この数字が Marigold V2 で幾らになるかが、方針の生死を決める。**

---

## 4. 一貫性のために必ず揃えること

過去に 2 度刺されているので明示する。

| | 内容 | 過去の事故 |
|---|---|---|
| **dtype** | `np.save(p, pred.astype(np.float16))` で保存し、採点時に `.astype(np.float16).astype(np.float64)` で読む | キャッシュを作った run は fp32 で採点、後の run は fp16 往復を採点して不一致 |
| **フレーム集合** | NYUv2 **654 枚すべて**。欠けたら `--require_masks_in` で明示的に揃える | 物体マスク 521 枚と全体 654 枚を混ぜ、baseline が 0.0726 と 0.0693 に割れた |
| **採点コードは 1 つ** | Marigold V2 側に eval を複製しない。予測 `.npy` を本リポジトリに渡して採点する | — |
| **fill_radius** | 3 固定。変えるなら必ず掃引する | 7 を掃引せずに選び、結論 3 件を撤回した |

---

## 5. 見積もり

| Phase | 作業 | 時間 |
|---|---|---|
| 0 | 推論 654 枚 ＋ aligner 実装 ＋ ゲート確認 | 2〜3 時間 |
| 1 | BF1 測定 | 15 分 |
| 2 | 鋭化 ＋ 対照 | 30 分 |
| 3 | オラクル上限 | 30 分 |

**合計 4 時間程度。学習は一切しない。**

最大の不確実性は Phase 0 のゲートで、`Log-stage2` の出力仕様（値域・符号の向き・
log の底・eps）を取り違えると abs_rel が再現しない。そこで止まる可能性が一番高い。

代替案として `depth/Disparity-base` チェックポイントもある（アフィン不変**逆数深度**を出すので
既存の `align_to_gt` がそのまま使える）。ただし Stage 1 recipe の 30k step で精度は本命より低く、
論文の表と直接比較できない。**Phase 0 が 2 回失敗したらこちらへ退避する。**

---

## 6. この測定がもたらすもの

| 結果 | 修論への影響 |
|---|---|
| 鋭化が Marigold V2 でも効く | 主張が「現行 SOTA の改善」になる。**最良** |
| 効かないが上限は高い | 手法は要るが器具が足りない。学習側に投資する根拠 |
| 効かず上限も低い | この方針は畳む。**ただし「土台の進化が境界問題を解いた」ことを測定で示したのは結果である** |

3 番目でも書けるものが残る点は重要である。
