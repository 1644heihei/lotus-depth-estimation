# Object Attention Conditioning 実装レビュー

全ファイルを精査しました。**致命的なバグ（動かない・クラッシュする系）は見当たりません**。実装は計画通りに正しく繋がっています。

ただし、**精度が向上しない根本原因と考えられる設計・実装上の問題点**が複数あります。重要度順に列挙します。

---

## 🔴 精度が上がらない主要因（高影響）

### 1. Object token に空間位置情報が不足している

> [!IMPORTANT]
> これが最大の問題と考えられます。

現在の `ObjectAttentionEncoder` は各物体の bbox 座標 `(cx, cy, w, h)` をスカラー特徴量として渡していますが、**UNet の cross-attention は「どの空間位置にどのトークンが関連するか」を知る仕組みがありません**。

CLIP の text token はテキスト全体に等しく影響を与えますが、object token は「画像の特定の領域」に影響を与えるべきです。しかし cross-attention には**位置バイアスもSpatial Gatingもない**ため、UNet のすべての spatial location が全 object token を等しく attend してしまいます。

結果として:
- 右上の物体の深度情報が、左下の画素にも影響する
- UNet がこの不整合を無視するように学習してしまう（＝token を使わない方が安全）

**改善案:**
```
方法A: Gated Cross-Attention（Flamingo 方式）
  - UNet の各 cross-attention block の後に、object 専用の
    gated cross-attention layer を追加
  - tanh gate をゼロ初期化して、初期は object を無視
  
方法B: Spatial Attention Bias
  - 各 object token の bbox を latent 空間にマッピング
  - attention score に position-dependent bias を加算
  - bbox 内の位置に高い bias、外部には 0 or 負のバイアス

方法C: Spatial Token Map（ControlNet 風）
  - object 情報を空間的なマップとして UNet に注入
  - cross-attention ではなく、add/concat で空間条件付け
```

### 2. Encoder の表現力が不足している可能性

[`ObjectAttentionEncoder`](file:///d:/lotus/lotus-depth-estimation/utils/object_attention_condition.py#L282-L317) は 2層 MLP（`hidden_dim=256`）のみで、14 次元の特徴量 + 64次元の class embedding → 1024次元の cross-attention token に射影しています。

```python
self.projector = nn.Sequential(
    nn.Linear(class_embed_dim + continuous_dim, hidden_dim),  # 78 → 256
    nn.SiLU(),
    nn.Linear(hidden_dim, cross_attention_dim),              # 256 → 1024
    nn.LayerNorm(cross_attention_dim),
)
```

この 2 層では、複雑な物体・深度関係を CLIP の 1024 次元空間に適切にマッピングするには浅すぎます。

**改善案:** 3-4 層にする、residual connection を追加、hidden_dim を 512 に増やす。

### 3. 入力特徴量の情報量が制限的

[`detection_attention_features`](file:///d:/lotus/lotus-depth-estimation/utils/object_attention_condition.py#L52-L81) の 14 次元特徴:

| 特徴 | 問題 |
|------|------|
| `cx, cy, w, h` | bbox の粗い位置のみ |
| `log(area)`, `log(aspect)` | `w,h` と冗長 |
| `score` | 検出信頼度（深度には間接的） |
| `mask_*` (6 feat) | マスク形状統計（深度への寄与は小さい） |
| `log(pred_depth_m)` | **唯一の深度関連特徴** |

**問題**: 14 次元中、深度に直接有用な情報は `log(pred_depth_m)` の 1 次元だけ。bbox座標 + マスク統計は「どこに何があるか」の補助情報に過ぎません。regressor の予測深度 1 個のスカラーだけで、UNet が「その物体周辺の深度をどう改善すべきか」を学ぶのは困難です。

---

## 🟡 精度改善を妨げている可能性がある設計（中影響）

### 4. 条件 Dropout が高すぎる可能性

学習スクリプトで `--object_attention_dropout_p=0.2` を使用しています。つまり 20% の確率で**バッチ全体**の object token が無効化されます。

8000 step × 0.2 = 1600 step 分が object token なしで学習されることになります。8000 step という短い学習ではこのドロップアウトが高い可能性があります。

**改善案:** dropout を 0.05〜0.1 に下げる、または学習初期は 0 にして後半で上げる。

### 5. 学習 step が少なすぎる可能性

[学習スクリプト](file:///d:/lotus/lotus-depth-estimation/train_scripts/train_lotus_d_object_attention_4ch.ps1#L41) で `--max_train_steps=8000` は、新しいモジュール（encoder）と UNet の共同 fine-tune には短い可能性があります。特に cross-attention を通じた間接的な条件付けは、直接的な入力チャネル連結より収束が遅い傾向があります。

### 6. UNet LR と Encoder LR が異なるが、比率が不明確

```
--learning_rate=1e-5     # UNet
--object_encoder_lr=1e-4 # Encoder (10倍)
```

Encoder は新規モジュールなので高い LR は理にかなっていますが、UNet 側の 1e-5 が低すぎて cross-attention 重みが十分に更新されていない可能性があります。

### 7. Warmup なし (`--lr_warmup_steps=0`)

新しい encoder のランダムな出力が学習初期に UNet の cross-attention に悪影響を与える可能性があります。warmup を入れるか、encoder のゼロ初期化を検討すべきです。

---

## 🟢 実装は正しいが確認した箇所（低影響）

### ✅ Token 連結と attention mask は正しい

[`append_object_attention_tokens`](file:///d:/lotus/lotus-depth-estimation/utils/object_attention_condition.py#L336-L356) は、CLIP の text token に object token を seq 方向に正しく連結し、combined attention mask も正しく生成しています。padding token は mask = False で softmax から除外されます。

### ✅ 学習と推論の経路は一致している

- 学習: `train_lotus_d.py` L1757-1802 で `encode_object_attention_condition` 呼び出し
- 推論: `pipeline.py` L1154-1181 で同じ関数を使用
- テスト `test_cache_and_online_condition_are_identical` で cache ↔ online の一致を検証

### ✅ RGB reconstruction ブランチでの object token 処理は正しい

`--disable_rgb_reconstruction` で RGB 復元は無効化されています。有効時も、RGB ブランチでは `object_mask` を全ゼロにして正しく処理しています。

### ✅ Pipeline の保存・復元は正しい

`object_condition_encoder` は `_optional_components` に登録され、`register_modules` で保存、checkpoint の save/load hook で正しく処理されています。

---

## 推奨する次のステップ

| 優先度 | アクション | 期待される効果 |
|--------|------------|----------------|
| 1 | Spatial attention bias の追加 | 物体位置と深度予測の空間的対応を確立 |
| 2 | Encoder を 3-4 層に深くする | 特徴表現力の向上 |
| 3 | Dropout を 0.05 に下げる | 限られた step で token 利用を最大化 |
| 4 | 学習 step を 20,000 に増やす | 収束の余地を確保 |
| 5 | Warmup 500 steps 追加 | 学習初期の安定性向上 |

> [!WARNING]
> 最も根本的な問題は **#1 の空間位置情報の欠如** です。これを解決しないと、他の改善策の効果は限定的です。Cross-attention token だけでは「どこの画素をどう変えるか」が伝わりません。
