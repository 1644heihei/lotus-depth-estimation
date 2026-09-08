# 輪郭の入力チャンネル条件付け — 結果

実施: 2026-09-07 〜 09-08
前提: [`contour_sharpening_findings.md`](contour_sharpening_findings.md)
ブランチ: `feature/text-conditioning`

---

## 0. 結論

**効かなかった。判定基準 3 つすべて不成立。**

9 通り（3 モデル × 3 入力）がすべて BF1 0.0470〜0.0474 に収まり、幅は 0.9%。
学習時の条件も推論時の入力も結果に影響していない。
そして全条件が学習前の 0.0693 から **−32%**。学習税が支配している。

輪郭は**届いており、出力を 0.2% 動かし、輪郭付近に局在する**。
しかしその局在は畳み込みの機械的な帰結で、学習が獲得した使い方ではない
（同じ局在が、輪郭を一度も見ずに学習した対照でも起きる）。
**読んだが意味を汲まなかった**、が正確な記述である。

---

## 1. 何を試したか

[`contour_sharpening_findings.md`](contour_sharpening_findings.md) で、輪郭を障壁として
深度を埋め直す後処理に **+279.9% BF1**（対照補正済み、学習ゼロ、abs_rel も改善）の
上限があることを測った。届かない理由は器具の精度で、
後処理は輪郭の位置を盲信するため **1.71px** を要求し、SAM は 2.00px しか出せない。

**学習だけがこの 0.29px を埋めうる。** 後処理にできないこと —— 「この辺に段差がある」という
近似情報として輪郭を読むこと —— を学習は獲得できる可能性がある。

入力チャンネルとして与える経路が成立することは、着手前に 2 点測って確認した。

| 前提 | 測定値 |
|------|--------|
| 輪郭は VAE を通過するか | **91.1% が画素完全、99.8% が 1.71px 以内**で復元 |
| ゼロ初期化で税が回避できるか | `expand_unet_conv_in(zero_init=True)` は step 0 で事前学習モデルと厳密に一致 |

---

## 2. 設定

| run | 学習時の輪郭チャンネル | 排除するもの |
|-----|---------------------|-------------|
| **C-null** | 常に空（−1） | チャンネル追加そのものの効果と、学習 3000 step の効果 |
| **C-cont** | この画像の SAM 輪郭 | 本命 |
| **C-shuf** | **他画像の輪郭** | 「輪郭らしい入力なら何でも効く」の可能性 |

共通: LoRA `all` rank 8 / alpha 16 / LR 1e-5 cosine / warmup 200 / batch 8 / res 512 /
3000 step / seed 42 / Hypersim hold-out 20 シーン除外 / 各 82 分。

輪郭は SAM (ViT-H) に YOLO の bbox をプロンプトとして生成（Hypersim 59,543 枚、
うち 61.5% に輪郭）。VAE で符号化して 4ch を conv_in に追加（4ch → 8ch）。

評価は NYUv2 654 枚。**3 モデルそれぞれを 3 つの入力で測る**ので 9 通り。

---

## 3. 結果

| 学習時 | 評価入力 | BF1 | abs_rel | delta1 | recall ON | recall OFF |
|--------|---------|----:|--------:|-------:|----------:|-----------:|
| **学習前 Lotus** | — | **0.0693** | **0.05000** | 0.9711 | 47.7% | **23.9%** |
| C-null | none | 0.0472 | 0.05826 | 0.9655 | 33.3% | 17.2% |
| C-null | contour | 0.0474 | 0.05817 | 0.9656 | 33.5% | 17.2% |
| C-null | shuffled | 0.0474 | 0.05821 | 0.9655 | 33.5% | 17.3% |
| C-cont | none | 0.0471 | 0.05834 | 0.9654 | 33.2% | 17.1% |
| **C-cont** | **contour** | **0.0473** | 0.05823 | 0.9655 | 33.4% | 17.2% |
| C-cont | shuffled | 0.0473 | 0.05827 | 0.9654 | 33.4% | 17.2% |
| C-shuf | none | 0.0470 | 0.05834 | 0.9655 | 33.2% | 17.1% |
| C-shuf | contour | 0.0471 | 0.05824 | 0.9655 | 33.4% | 17.1% |
| C-shuf | shuffled | 0.0472 | 0.05828 | 0.9654 | 33.4% | 17.2% |

### 判定（着手前に固定）

| 比較 | 差 | 判定 |
|------|---:|------|
| C-cont/contour > C-shuf/shuffled | **+0.0001** | **不成立** |
| C-cont/contour > C-null/none | **+0.0001** | **不成立** |
| C-cont/contour > 学習前 0.0693 | **−0.0220（−31.8%）** | **不成立** |

---

## 4. 「使わなかった」のか「使ったが変わらなかった」のか

区別できる。3 つ測った。

### 4.1 重みは繋がっているが 2 桁小さい

| | RGB チャンネル \|w\| 平均 | 輪郭チャンネル \|w\| 平均 | 比 |
|---|--------------------:|---------------------:|----:|
| C-null | 4.2155e-02 | 4.4986e-04 | **0.0107** |
| C-cont | 4.2155e-02 | 4.6597e-04 | **0.0111** |
| C-shuf | 4.2155e-02 | 4.4921e-04 | 0.0107 |

ゼロではないので「完全な無視」ではない。ただし RGB の **約 1%**。
そして **C-cont と C-null がほぼ同じ**。中身が意味を持つなら C-cont だけ大きくなるはずである。

### 4.2 出力は動く。0.2%

輪郭を与えた場合と空の場合で、深度予測は **0.20〜0.23%** 動く。
確かに使われている。しかし C-cont（0.2254%）と C-null（0.2049%）の差は小さく、
**輪郭を一度も見ずに学習したモデルも同じくらい動く**。

### 4.3 変化は輪郭付近に集中する。ただし対照も同じ

| | 輪郭近傍の \|diff\| | 遠方 | 比 |
|---|-----------------:|-----:|----:|
| C-cont | 0.00167 | 0.00077 | **2.16 倍** |
| C-null | 0.00154 | 0.00070 | **2.20 倍** |

変化は輪郭の近くに 2 倍強く出る。位置情報が空間的に届いている証拠ではある。

**しかし C-null が同じ 2.20 倍を示す。** C-null は学習中に一度も輪郭を見ていない。
それでも輪郭を入力すると同じように輪郭付近が動く。
conv_in は空間畳み込みなので、入力に線が入れば出力もそこで動く —— それだけである。

**したがってこの局在は学習の成果ではない。**

---

## 5. 学習税はゼロ初期化でも避けられなかった

全条件が学習前から **−32%**。
テキスト条件付けの T-null で測った −37.4% とほぼ同じ。

ゼロ初期化は **step 0 の税**をゼロにするが、3000 step 後には LoRA 側の適応が同じだけ削る。
「触る範囲を絞れば害が小さい」は、[`text_conditioning_results.md`](text_conditioning_results.md)
の attn2 限定 LoRA（−15.7%、深度帯 LoRA の −4.6〜−12.6% より大きい）に続いて
**2 度目の反証**である。

---

## 6. 実装の失敗と、その教訓

**1 回目の学習 3 本（約 4 時間）は完全に無効だった。**

```
1344: expand_unet_conv_in(unet, extra_in_channels=4, zero_init=True)  # 新チャンネルをゼロで初期化
1369: unet.requires_grad_(False)                                       # 全体を凍結
1395: unet.add_adapter(lora_config)                                    # attention の射影だけ学習可能に
```

**新チャンネルの重みはゼロのまま凍結され、LoRA の対象にも入らない。**
輪郭は VAE で符号化され UNet に渡され、そしてゼロを掛けられて消えていた。
保存された conv_in を検査して初めて分かった（`extra slices abs max = 0.000e+00`）。

煙試験では `[contour sample] on-contour frac=[0.0, 0.0012, ...]` を確認していた。
**ログは「テンソルを渡した」ことしか証明しない。**

### 対処

`utils/expanded_conv_in.py` を新設し、整合すべき 3 つを 1 箇所に置いた。

| 責務 | 内容 |
|------|------|
| 凍結解除 | `add_adapter` の**後**に `conv_in` を学習可能化 |
| 保存 | `save_lora_adapter` は attention しか書かないので `conv_in.safetensors` を隣に保存 |
| 読み込み | 評価はベースモデル（4ch）から作るので、アダプタ適用**前**に拡張と復元 |

そして**コードの正しさに依存しない検査**を学習終了時と評価開始時の両方に置いた。

```python
if extra_channel_energy(unet, 4) == 0.0:
    raise RuntimeError("the contour channel's weights are still exactly zero")
```

単体テスト 14 件（`tests/test_contour_conditioning.py`）は、バグそのものをテストとして記述している。

```python
def test_zero_slices_make_the_extra_channels_inert(self):
    a = u(cat([img, empty_contour]))
    b = u(cat([img, random_contour]))
    torch.testing.assert_close(a, b)   # ゼロ重みでは同一になる
```

2 回目の学習では 3 本とも `|w|max` が 1.81〜1.86e-03 まで動き、機構が働いたことを確認済み。
**本書の否定的な結果は、実装の不備によるものではない。**

---

## 7. 物体検出という方針の総括（更新）

| 経路 | 結果 | 出典 |
|------|------|------|
| 物体の深度**レベル** | 正味 −0.60%（対照以下） | [`object_oracle_control.md`](object_oracle_control.md) |
| 物体の**内部形状** / **周辺** | net/area が背景と同水準 | [`object_band_decomposition.md`](object_band_decomposition.md) |
| 物体の**境界位置**（後処理・YOLO） | 正味 −6.6% | [`object_boundary_closure.md`](object_boundary_closure.md) |
| 物体の**境界位置**（後処理・SAM） | 正味 +4.2%、上限の 1.5% | [`contour_sharpening_findings.md`](contour_sharpening_findings.md) |
| RGB エッジによる位置補正 | 全設定で悪化 | [`rgb_edge_route_closure.md`](rgb_edge_route_closure.md) |
| 物体名（テキスト、学習なし / あり） | 内容非依存 / 対照より悪化 | [`text_conditioning_results.md`](text_conditioning_results.md) |
| **輪郭（入力チャンネル、学習あり）** | **9 通りすべて同一、学習前から −32%** | **本書** |

**後処理の上限 +279.9% は本物のまま残っている。** 閉じたのは「その上限に到達する手段」であり、
学習もその手段にはならなかった。

---

## 8. 成果物

| パス | 内容 |
|------|------|
| `utils/contour_condition.py` | 輪郭マップの構築（マスクは元解像度で復号し、輪郭を学習解像度へリサイズ） |
| `utils/expanded_conv_in.py` | 拡張 conv_in の凍結解除・保存・読み込みと検査 |
| `tests/test_contour_conditioning.py` | 14 件。出力が変わることを直接検証 |
| `build_sam_masks.py` | SAM マスク生成（`--dataset hypersim --union_only`） |
| `train_scripts/run_contour_condition.ps1` | 学習 3 本 |
| `eval_contour_condition.py` | 3 モデル × 3 入力の評価 |
| `output/lora-contour-{null,cont,shuf}/` | 学習済み LoRA + conv_in |
| `output/eval_contour_{null,cont,shuf}/` | 上表 |
| `D:/lotus/data/hypersim_sam_seg/` | Hypersim 59,543 枚の SAM マスク |
