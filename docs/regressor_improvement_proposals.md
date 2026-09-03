# Object Depth Regressor 精度向上案

## 現状の把握

### Regressor v4_mask の構成

| 項目 | 値 |
|------|-----|
| アーキテクチャ | `ObjectDepthMLP`: Embedding(91, 32) + 3層MLP [256→128→64] → log_depth |
| 入力特徴量 | 13 次元（feature_version=3: bbox 7 + mask 6） |
| 学習データ | Hypersim train のみ（86,551 物体, scene-level split） |
| 損失関数 | SmoothL1Loss（log_depth 空間） |
| Holdout 精度 | **abs_rel=0.318, δ1=0.432, RMSE=1.85m** |

### 問題の本質

Regressor は「YOLO bbox + マスク形状 → 物体深度」を予測するタブラ回帰器です。
**abs_rel=0.318** は「予測深度が GT に対して平均 32% ズレる」ことを意味し、
この不正確な深度情報を Attention token として UNet に渡しても、ノイズにしかなりません。

---

## 改善案（優先度順）

### 🔴 1. 画像特徴量の導入（最大のインパクト）

**現状**: Regressor は bbox 座標・サイズ・マスク統計のみ（画像を見ていない）。
同じサイズの「椅子」でも近い/遠いシーンでは深度が全く異なりますが、
bbox 特徴だけではその区別がつきません。

**案A: ROI 特徴量の追加（最も実装が楽）**

```
YOLO bbox → ROI crop → 小さいCNN（ResNet-18 の最終層など） → 特徴ベクトル
→ 既存の bbox 特徴と concat → MLP → log_depth
```

- ROI 内の **テクスチャ・明度・色合い** は深度の手がかりになる（暗い/ぼやけた物体は遠い傾向）
- ImageNet 事前学習の CNN を凍結して特徴抽出するだけなら追加学習コストは小さい

**案B: Depth Anything / DPT の特徴量を流用**

- 公式 Lotus-D の VAE encoder output や、Depth Anything v2 の中間特徴を ROI ごとに抽出
- 「既に深度を推定できるモデル」の特徴量を使うため、非常に情報量が高い
- ただし推論時に追加モデルのフォワードパスが必要

**期待効果**: abs_rel を 0.32 → **0.15〜0.20** 程度に改善可能（画像を見れるだけで大幅改善）

---

### 🔴 2. 学習データの拡充

**現状**: Hypersim train のみ（合成データ、室内シーン限定）。
NYUv2 は実写の室内シーンで、Hypersim とドメインが異なります。

**案A: 追加データセットの投入**

| データセット | 特徴 | 物体数の目安 |
|-------------|------|-------------|
| ScanNet train | 実写室内、深度あり | 数万〜 |
| SUN RGB-D | 実写室内、多様な部屋 | 数万〜 |
| KITTI (自動車) | 屋外シーン | 数万 |

```
Hypersim train + ScanNet train + SUN RGB-D で学習
→ NYUv2 test で評価（データリーク なし）
```

**案B: Hypersim 内のデータ拡張**

- bbox に位置/サイズジッタを加える（±5〜10%）
- 深度に軽いノイズを加える（log空間で σ=0.05 程度）
- class_id を同カテゴリ内でシャッフル（まれに）

**期待効果**: ドメインギャップの縮小で abs_rel を **5〜15%** 改善

---

### 🟡 3. アーキテクチャの改善

**案A: MLP の拡大**

現状の [256→128→64] は小さい。

```python
# 提案
hidden = [512, 256, 128, 64]
embed_dim = 64
dropout = 0.1
```

**案B: 物体間コンテキストの導入（Transformer）**

同一画像内の複数物体の相対関係（「テーブルの上に置かれたコップ」）は深度の重要な手がかり。

```
各物体 → embedding → Self-Attention（画像内の全物体）→ 深度予測
```

- 現状は各物体を独立に予測しているが、同一シーンの他の物体情報があれば精度向上が見込める
- 実装はやや複雑（可変長入力のバッチ処理が必要）

**案C: 損失関数の改善**

```python
# 現状: SmoothL1Loss(log_pred, log_gt)
# 提案: 複合損失
loss = SmoothL1Loss(log_pred, log_gt) 
     + 0.5 * (abs(exp(log_pred) - gt) / gt).mean()  # abs_rel直接最適化
```

**期待効果**: 単独では abs_rel **3〜8%** 改善

---

### 🟡 4. 学習手法の改善

**案A: LR スケジューリング**

現状は AdamW + 固定 LR + Early stopping。
→ Cosine annealing + warmup で安定した収束を目指す。

**案B: エポック数増加 + 重み減衰調整**

```
epochs: 80 → 200
lr: 1e-3 → 3e-4（warmup 10 epochs）
weight_decay: 1e-4 → 1e-3
```

**案C: クラス不均衡対策**

Hypersim は「椅子」「テーブル」に偏っている可能性がある。
稀なクラスのサンプルを重み付けすることで、全クラスの精度を底上げ。

**期待効果**: abs_rel **2〜5%** 改善

---

### 🟢 5. 特徴量エンジニアリング

**案A: 相対特徴量の追加**

```python
# 同一画像内の他物体との相対位置
relative_depth_rank  # この物体は画像内で何番目に手前か
num_objects_in_scene # 画像内の物体数
```

**案B: bbox 比率特徴の拡充**

```python
# 画像全体に対する相対位置
bbox_cx / image_width   # 既存
bbox_cy / image_height  # 既存
bbox_area / image_area  # 画像に対する占有率
aspect_ratio_image      # 画像自体のアスペクト比
```

---

## 推奨アクション（優先順位）

| 優先度 | アクション | 工数 | 期待改善 |
|--------|-----------|------|---------|
| **1** | ROI CNN 特徴量の追加（案1A） | 中 | abs_rel -30〜40% |
| **2** | ScanNet/SUN RGB-D データ追加（案2A） | 中 | abs_rel -5〜15% |
| **3** | MLP 拡大 + 損失関数改善（案3A+3C） | 小 | abs_rel -3〜8% |
| **4** | Hypersim データ拡張（案2B） | 小 | abs_rel -3〜5% |
| **5** | LR スケジュール改善（案4A+4B） | 小 | abs_rel -2〜5% |

> [!IMPORTANT]
> **最も効果的なのは案1（画像特徴量の導入）** です。
> 現在のRegressorは「画像を一切見ずに」深度を予測しています。
> これは人間が「目を閉じて、"幅30cm高さ50cmの椅子がある"と言われて距離を当てる」のと同等です。
> 画像を見れば（テクスチャ、明度、遠近法的手がかり）、精度は大幅に向上します。

> [!TIP]
> ただし、Regressor 精度の向上が最終的な Lotus-D パイプラインの精度に直結するかは
> 前回のレビューで指摘した **spatial attention** の問題次第です。
> Regressor 改善と並行して、Attention token → UNet への情報伝達経路の改善も進めるべきです。
