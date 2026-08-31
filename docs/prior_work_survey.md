# 先行研究調査 — 疎な深度点による手法は先行されている

実施: 2026-09-01
ブランチ: `feature/object-depth-aux-loss`
前提: [`sparse_points_findings.md`](sparse_points_findings.md)

疎な深度点が誤差の 43〜76% を回復すると測定できたため、
モデル側の手法（方針 A）に進む前に先行研究を調査した。

**結論: 方針 A として設計した構成は、2026 年 5 月の論文とほぼ同一である。**

---

## 0. 要約

| 項目 | 判定 |
|------|------|
| 疎な点を使う depth completion 手法 | **完全に先行されている** |
| 単一ステップ拡散 + zero-init + 疎な点という設計 | **完全に先行されている**（Marigold-SSD） |
| 補間ベースラインとの比較 | **実施済み**（先方の Fig. 7） |
| 疎度レベルを変えた評価 | **実施済み**（#500〜#15360） |
| **誤差の大域/局所分解** | **見当たらない — 独自** |
| **後処理 3 経路の閉塞の実証** | **見当たらない — 独自** |
| **物体・材質条件付けの系統的否定** | **見当たらない — 独自** |

---

## 1. この領域の混雑状況

| 論文 | 会議 / 年 | 内容 |
|------|-----------|------|
| **Marigold-DC** | **ICCV 2025** | 拡散深度モデル + 疎な点を test-time guidance で注入。学習不要。ETH Zürich |
| **Marigold-SSD（Need for Speed）** | 2026-05 arXiv | **単一ステップ**拡散 + 条件付きデコーダ。DTU + ETH Zürich |
| Large Depth Completion Model | **ICLR 2026** | 疎な観測からの大規模モデル |
| OASIS-DC | **ICRA 2026** | 単眼疑似深度と疎な点の出力レベル整合 |
| Propagating Sparse Depth via Depth Foundation Model | **IEEE TIP 2025** | 分布外での depth completion |
| Depth Anything with Any Prior | 2025 | DA に任意の事前情報 |
| DenseFormer | 2025 | 拡散による sparse-to-dense |

`Awesome-Foundation-Model-Based-Depth` というキュレーションリストが存在するほど活発。

---

## 2. Marigold-SSD との照合（精読結果）

> Gregorek et al., *Need for Speed: Zero-Shot Depth Completion with Single-Step Diffusion*,
> arXiv:2603.10584v2 (2026-05-02), DTU + Pioneer Centre for AI + ETH Zürich

| 設計要素 | 本プロジェクトの方針 A | **Marigold-SSD** |
|----------|----------------------|------------------|
| ベースモデル | 単一ステップ拡散深度（Lotus） | **Marigold-E2E**（単一ステップ） |
| 条件付け信号 | 疎な深度点 | **疎な深度点** |
| 構造 | 凍結バックボーン + **zero-init 側枝** | 条件付きデコーダ + **zero-init 畳み込み**（ControlNet 由来） |
| 凍結範囲 | UNet 凍結 | VAE エンコーダ凍結、条件付きデコーダ + UNet を学習 |
| 損失 | scale-shift 不変を提案 | **L1**（affine-invariant 損失を検討の上で不採用） |
| 疎度の扱い | 点数と回復率の関係を測定 | 学習時に密度を `[l%, h%]` で一様サンプル、**複数疎度で評価** |
| 学習コスト | — | **4.5 GPU 日** |
| 評価 | — | 屋内 4 + 屋外 2 ベンチマーク（ScanNet, iBims-1, VOID, NYUv2, KITTI, DDAD） |

**設計が偶然一致したのではなく、同じ制約（単一ステップ・凍結・zero-init）から
同じ結論に至っている。**

### 本プロジェクトの測定と重なる論点

先方の Fig. 7 は「commonly used sparsity level of 5000 points では、
高度なモデルが**単純な Barycentric 補間に負けうる**」ことを示している。

本プロジェクトが測った「RBF 補間で 20 点 +43%」は、まさに同じ論点であり、
**補間ベースラインの強さも既に指摘済み**。

### 先方が挙げている限界

> "Our end-to-end fine-tuning requires to set sampling density range of the condition.
> As demonstrated in ablation studies, completion of **out-of-distribution depth maps
> may exhibit a steep performance drop**."

学習時に想定した疎度から外れると性能が落ちる、という限界を自ら記載している。

---

## 3. 独自性が残っている部分

いずれも**手法ではなく測定・分析**の側にある。

| 内容 | 本プロジェクトでの結果 | 先行研究での扱い |
|------|----------------------|-----------------|
| 誤差の大域/局所分解 | 低周波が **Booster 75.3% / NYUv2 54.7%** | 見当たらない |
| 後処理の到達不能性 | **3 経路すべてを実測で閉塞**（固定バイアス / 摂動平均 / 学習補正、R² < 0） | 見当たらない |
| 「なぜ疎な点が効くのか」 | **誤差の 75% が低周波の大域成分だから** | 「効くから使う」であり、誤差構造からの説明は見当たらない |
| 物体・材質条件付けの否定 | Oracle 上限 2.1% / 5.3%（対照補正後） | 見当たらない |
| TTA の周波数分解による説明 | アンサンブルは**局所を 21.5%、大域を 3.7%** しか消さない | ensembling 自体は Marigold にあるが、この分解は見当たらない |

---

## 4. 判断

**方針 A（疎な点を使うモデル）は、このまま進めても新規性が立たない。**
4.5 GPU 日で学習した論文が 6 ベンチマークで評価済みであり、
補間ベースラインとの比較も疎度分析も済んでいる。

この分野は資源のある研究室（ETH Zürich, DTU）が 2025〜2026 年に連続で埋めており、
同じ土俵で競うのは現実的でない。

### 残る選択肢

| 案 | 内容 | 懸念 |
|----|------|------|
| **(1) 貢献を分析に置き直す** | 誤差構造の解明と各介入の到達可能性の定量化。手法は既存を使う | 「調査研究では通らない」という制約に抵触。**指導教員との相談が必須** |
| **(2) 別の追加情報を探す** | カメラパラメータ、重力方向、複数視点など | 調査してから決めるべき。既に埋まっている可能性 |
| **(3) 応用条件に絞る** | 実センサの走査線パターン、極端に少ない点数、リアルタイム制約 | Marigold-SSD が limitations に挙げる**分布外疎度での性能低下**は隙になりうる |

> [!IMPORTANT]
> **方針転換の判断は指導教員と行うこと。** 本調査結果（特に Marigold-SSD との重複）は
> そのまま共有する価値がある。要求水準・残り時間・専攻の慣行は指導教員が判断すべき事柄。

---

## 5. 参照

- Marigold-DC: https://arxiv.org/abs/2412.13389 （ICCV 2025）/ https://github.com/prs-eth/Marigold-DC
- Need for Speed（Marigold-SSD）: https://arxiv.org/pdf/2603.10584 / https://dtu-pas.github.io/marigold-ssd/
- Large Depth Completion Model: https://arxiv.org/html/2605.30115v1 （ICLR 2026）
- Propagating Sparse Depth via Depth Foundation Model: https://arxiv.org/abs/2508.04984 （IEEE TIP 2025）
- Awesome-Foundation-Model-Based-Depth: https://github.com/Ideal-111/Awesome-Foundation-Model-Based-Depth
