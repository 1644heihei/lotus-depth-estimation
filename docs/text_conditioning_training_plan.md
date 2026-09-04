# テキスト条件付けの学習 — 実装計画

作成: 2026-09-04
前提: [`rgb_edge_route_closure.md`](rgb_edge_route_closure.md) / [`object_boundary_closure.md`](object_boundary_closure.md)

---

## 0. 計画が立脚する測定値

| 事実 | 値 | 出典 |
|------|----|------|
| 真の深度不連続のうち画像にエッジがないもの | **54.6%** | `eval_edge_dependence.py` |
| そのうち Lotus が拾えるもの | **10.5%**（エッジ上なら 31.2%） | 同上 |
| テキスト→領域の結合の残存度 | **1.9%**（lift 1.079 / 完全なら 5.18） | `eval_cross_attention_localization.py` |
| 学習なしのテキスト条件付け | 内容非依存（generic ≥ classes、位置語も無効） | `eval_text_prompt_conditioning.py` |
| fine-tune 税（実測） | 同一パラメータ数で **−4.6〜−12.6%** | [`adaptation_placement_findings.md`](adaptation_placement_findings.md) |
| 学習コスト（実測） | **1.34 s/step** → 1000 step = 22 分 | `output/lora-placement-*` の checkpoint 時刻 |

**狙い**: 画像に写らない深度段差を、物体の意味情報で補えるようにする。
そのためにはテキスト→領域の結合を作り直す必要があり、それは学習でしかできない。

### 位置情報をどう入れるか — B + C を併用する

| 案 | 位置の出どころ | 学習が担うもの | 判定 |
|----|--------------|--------------|------|
| A. プロンプトに位置語 | 3×3 セル（480×640 で約 160px 角） | 「bottom centre」が配置を意味することの学習 | **採らない**。粒度が必要精度（約 1px）と桁違い |
| **B. 名前のみ** | **学習の結果として獲得** | テキスト→領域の結合をゼロから再構築 | **採る** |
| **C. 空間バイアス** | **検出の bbox から与える（既知）** | 領域に固定済みの意味トークンの使い方 | **採る** |

C の要点は、**位置を学習させないこと**。位置は検出結果、意味は CLIP の事前学習済み埋め込み。
今日測った最大の障害（結合が 1.9% まで潰れている）を、学習させずに手渡す。

> **bbox かマスクか**: 当初「マスクから与える」と書いたが、`hypersim_sem_masks` は
> 全物体を統合した 1 枚で、インスタンス別ではない。クラスごとの領域を得るには
> YOLO-seg を 59,544 枚に再実行する必要がある。
> 一方バイアスが効くのは潜在解像度（res 512 学習なら 64×64 以下）で、bbox と
> シルエットの差は数セル程度に潰れる。**第 1 サイクルは bbox で進める**。
> C に見込みが出た段階で、統合マスクと bbox の積によるインスタンス近似を検討する。

B と C を両方入れ、**判定基準 (d) を「バイアス OFF での lift」で測ることで分離する**。
バイアス ON のまま lift を測ると C の効果が支配してしまい、B が働いたかが見えない。

**14px のずれについて**: 後処理実験ではマスク輪郭の**その画素**に段差を置いたので 14px 外すと
偽陽性になった。空間バイアスは「この近辺に物体境界がある」と伝えるだけで、正確な位置は
モデルが画像から決める。加えて 768 推論なら潜在は 96×96（8 倍ダウンサンプル）なので、
画像空間の 14px は**潜在空間で約 1.75 セル**（学習は res 512 なので更に粗い）。
バイアスは潜在解像度で効くため、画素空間で見たずれほどには害しない。

**前例の注意**: 本リポジトリは空間バイアスを既に試して失敗している
（[`object_attention_training_results.md`](object_attention_training_results.md)）。
ただしそのときトークンの中身は**学習した regressor 特徴量**で、意味をゼロから獲得する
必要があった。今回は中身を **CLIP の事前学習済み埋め込み**に置き換える点が違う。
それでも「失敗した構成に近い」ことはリスクとして記録しておく。

---

## 1. 既に揃っているもの

| 資産 | 状態 |
|------|------|
| Hypersim の YOLO 検出 | `D:/lotus/data/hypersim_yolo_detections` に **59,544 件**、`class_name` 付き |
| 検出の読み出し | `utils/object_detection_cache.load_detections(path, root)` |
| 学習バッチ中の画像パス | `batch["image_pathes"]`（`train_lotus_d.py:1737` で既に使用） |
| LoRA の対象指定 | `--lora_target_blocks`（正規表現、マッチ 0 件で例外停止） |
| 評価スクリプト | `eval_text_prompt_conditioning.py`（6 プロンプト × 4 指標、対照込み）<br>**ただし LoRA を読めない → 2.5 で追加が必要** |
| 機構の検証 | `eval_cross_attention_localization.py`（lift + 2 対照）<br>**同上** |

**Hypersim の検出統計**（400 件標本）: 平均 1.60 個/画像、**39.5% は検出ゼロ**、クラス 27 種。
検出ゼロの画像は自然に空プロンプトになるので、暗黙のプロンプトドロップアウトとして働く。

---

## 2. コード変更（5 箇所）

### 2.1 サンプルごとのプロンプト — `train_lotus_d.py:1912-1923`

現状は全学習を通じて固定:

```python
prompt = ""
text_inputs = tokenizer(prompt, padding="do_not_pad", ...)
text_encoder_hidden_states = text_encoder(text_input_ids, ...)[0]
text_encoder_hidden_states = text_encoder_hidden_states.repeat(bsz, 1, 1)
```

変更点:

1. `batch["image_pathes"]` から `load_detections` でクラス名を引き、
   **評価と完全に同じ書式**で組む: `", ".join(sorted({d.class_name for d in dets if d.score >= thr}))`
   書式が食い違うと学習と評価が別条件になるので、共通関数に切り出して両方から呼ぶ。
2. `padding="do_not_pad"` → **`padding="max_length"` + `encoder_attention_mask`**。

   > **2026-09-04 に判明**: `LotusDPipeline.encode_prompt` の既定は
   > `padding_type="do_not_pad"` で、**推論時の系列長は実トークン数**（`"sink"` なら 3、
   > 空プロンプトなら 2）。バッチ化のために学習だけ 77 に固定すると、
   > **学習中は存在して推論時には無い 74 個のパディング埋め込みに cross-attention が向く**。
   > CLIP のパディング埋め込みはゼロではないので、これは実験全体を汚す
   > train/inference 不整合になる。
   >
   > 対処: 77 にパディングしたうえで **`encoder_attention_mask` で実トークンだけを有効化**する。
   > そうすれば実効的な内容が推論時の `do_not_pad` と等価になる。
   > `train_lotus_d.py` には `encoder_attention_mask` 変数が既にあり（現在 `None`）、
   > パイプラインも UNet へ渡しているので、配線は済んでいる。
   >
   > この不整合は空間バイアスの実装中に、幅 77 のバイアスが長さ 3 の key と
   > broadcast できず落ちたことで発見した。バイアスを入れなければ
   > 気付かないまま学習していた可能性が高い。

3. `.repeat(bsz, 1, 1)` を削除し、バッチ分をまとめてエンコード。
4. `--text_prompt_dropout_p`（既定 0.1）を追加。
   確率的に空文字列へ置換し、**テキストなしでも動くモデルを保つ**。
   これがないと、テキスト依存になって空プロンプト評価が壊れる。

`text_encoder` は凍結済み（`train_lotus_d.py:1258`）なので前向き計算のみ。
プロンプト文字列の種類は少ない（クラス 27 種の組合せ）ので、
**文字列をキーにした埋め込みキャッシュ**を持てば追加コストはほぼ消える。

### 2.2 cross-attention だけを対象にする LoRA — `train_lotus_d.py:1269-1283`

`--lora_target_blocks` に `"text"` を追加:

```python
"text": r".*attn2\.(to_q|to_k|to_v|to_out\.0)$",
```

`attn2` が cross-attention。`to_k` / `to_v` がテキスト側、`to_q` が画像側の射影で、
結合は q·k で決まるので両方要る。
`attn1`（自己注意）・resnet・畳み込みには触れないので、
深度推定の本体を動かさずに済む。これが fine-tune 税を抑える設計上の根拠。

UNet の `attn2` は 16 モジュール。実際のパラメータ数は起動時ログで確認する
（既存コードはマッチ 0 件なら例外で止まる）。

### 2.3 プロンプト構築の共通化

`utils/object_prompt.py` を新設し、以下を置く。

- `build_class_prompt(dets)` → プロンプト文字列
- `class_token_spans(tokenizer, prompt, dets)` → 各検出が対応するトークン位置

`train_lotus_d.py` と `eval_text_prompt_conditioning.py` の両方がこれを呼ぶ。
**学習と評価で書式がずれる事故を構造的に防ぐ。**
トークン位置の算出は `eval_cross_attention_localization.py::token_spans` に実装済みなので、
そこから移設して両者で共有する。

### 2.4 CLIP クラストークンへの空間バイアス（C）

既存の `utils/object_spatial_attention.py` は**追加トークン専用**で、
テキストトークン側の列はゼロで埋まっている:

```python
bias = object_bias.new_zeros(batch_size, 1, query_len, key_len)
bias[:, :, :, num_text_tokens:] = object_bias.transpose(1, 2).unsqueeze(1)
```

B + C ではクラス名を綴る**テキストトークンの列**に書く必要があるので、
`build_class_token_spatial_bias(...)` を新設する。

#### 受け渡しの構造

**バイアスは層ごとに作る。** 形状が `[B, 1, Q, 77]` で、Q は attention 層ごとに違うため
（res 512 学習なら 64² / 32² / 16² / 8²）、呼び出し側で 1 つ作って渡すことはできない。

既存実装と同じく、**入力を `cross_attention_kwargs` で渡し、processor 内部で層ごとに組む**:

| キー | 形状 | 内容 |
|------|------|------|
| `class_token_bbox` | `[B, K, 4]` | 正規化 (cx, cy, w, h) |
| `class_token_index` | `[B, K]` | その検出が対応する CLIP トークン位置 |
| `class_token_mask` | `[B, K]` | 実在する検出かどうか（パディング除け） |

`ObjectSpatialAttnProcessor` は既に `object_spatial_features` から層ごとに組む経路を
持っているので、そこに分岐を足す。`_attention_grid_size` もそのまま使える。

**K の扱い**: K は検出数。同一クラスの複数検出は**同じトークン位置を指すのでインデックスが
重複する**（scatter-max で集約する）。`--max_class_tokens`（既定 16）で上限を設け、
不足分は `class_token_mask=False` でパディングする。
Hypersim の実測は平均 1.60 個/画像・最大 9 個なので、16 で十分足りる。

#### 組み方

- 空間プロファイルは既存の `inside_score = exp(-0.5 * radial)` を再利用
- `inside_bias=10.0` / `outside_bias=-2.0` も既存値を初期値として踏襲
- **同一クラスが複数写る場合はトークン列上で max を取る**（「どれか 1 つがある場所を見てよい」）
- 「dining table」のようにクラス名が複数トークンにまたがる場合、全トークンに同じ値を書く
- **クラス名以外のトークン（BOS/EOS/区切り/パディング）の列はゼロのまま**

#### バイアスのドロップアウト（必須）

`--class_token_bias_dropout_p`（**既定 0.5**）を入れる。

T-both はバイアス ON で学習するため、**モデルがバイアスの存在を前提に適応してしまう**。
その状態で OFF 評価すると、「B が学習しなかった」のか「バイアスがなくて壊れた」のかを
区別できず、**判定基準 (d) の分離ロジックが成立しない**。
プロンプトドロップアウトと同じ発想で確率的にバイアスを切り、ON/OFF どちらでも
動くモデルにする。

**0.5 を選ぶ理由**: 0.1 ではモデルが学習の 90% でバイアスを見るため
「バイアスなしでも動く」の強制が弱く、OFF 評価の交絡が残る。
0.5 なら半分をバイアスなしで回すので分離が確実になる。
代償として C の学習信号は半分になり、**C 単独の効果は控えめに出る可能性がある**。
C に見込みが出た段階で、dropout を下げた 2 本目を回して効果の上限を測る。

#### 切り替え

`--class_token_spatial_bias`（既定 off）。**学習時と推論時で独立に指定する。**
判定基準 (d) がバイアス OFF での測定を要求するため。

#### 実装済み・検証済み（2026-09-04）

`build_class_token_spatial_bias` と `class_token_cross_attention_kwargs` を実装、
`ObjectSpatialAttnProcessor` に分岐を追加。単体テスト 16 件
（`tests/test_class_token_spatial_bias.py`: 形状 / ゼロ列 / max 集約 / 値域 / 行順不変）。

実機での配線確認（NYUv2 30 枚、公式モデル、`eval_cross_attention_localization.py`）:

| 領域 | バイアス OFF | バイアス ON |
|------|------------:|-----------:|
| そのトークン自身の物体領域 | 1.138 | **18.882** |
| 対照: 他クラスの領域 | 1.098 | 0.083 |
| 対照: 移動した領域 | 0.979 | 0.229 |

**位置を手渡す操作は意図どおり働く。** OFF 側は実装前と完全一致で、既存経路は無傷。

`class_token_num_text_tokens` は**渡さない**のが既定（実行時の系列幅を使う）。
上記の padding の件により、固定値を渡すと推論時に落ちる。

### 2.5 評価スクリプトに LoRA 読み込みを追加

**現状、`eval_text_prompt_conditioning.py` と `eval_cross_attention_localization.py` は
どちらも `LotusDPipeline.from_pretrained(core_model)` で公式モデルを読むだけで、
学習済み LoRA を読む経路がない。このままでは学習しても評価できない。**

`eval_regressor_predepth_nyuv2.py:284` に前例がある:

```python
detail_pipe.unet.load_lora_adapter(lora_dir, ...)
```

両スクリプトに `--lora_path`（`unet_lora/` ディレクトリ）を追加し、同じ経路で読む。
併せて `--class_token_spatial_bias` も追加し、推論時のバイアス ON/OFF を選べるようにする。

---

## 3. 学習する 3 本

| run | LoRA 対象 | 学習時のプロンプト | 空間バイアス | 目的 |
|-----|----------|------------------|------------|------|
| **T-null** | `text` | 常に `""` | off | **fine-tune 税の分離**。容量・step 数を揃えた対照 |
| **T-text** | `text` | クラス名、dropout 0.1 | off | **B 単独**（結合を学習で再構築できるか） |
| **T-both** | `text` | クラス名、dropout 0.1 | **on** | **B + C**（位置を手渡した場合） |

共通設定（`run_lora_depth_placement.ps1` に準拠）:
rank 8 / alpha 16 / LR 1e-5 cosine / warmup 200 / batch 8 / res 512 /
**3000 step**（checkpoint 500 ごと）/ seed 42

**学習中の validation は無効化する**（`--validation_steps=100000`、前例と同じ）。
サンプルごとのプロンプト化により validation が何を渡すかが曖昧になるうえ、
評価は checkpoint に対して後から統一条件で回すほうが条件を揃えやすい。

3000 step ≈ **67 分**/run。3 本で約 3 時間 20 分。

> T-null を必ず回すこと。これまでの実験は例外なく fine-tune 税を払っており、
> T-text 単独では「テキストが効いた」のか「LoRA が効いた/害した」のか分離できない。

## 4. 評価

既存スクリプトを学習済み LoRA に向けて実行する。

```
eval_text_prompt_conditioning.py      # empty / classes / shuffled / generic
eval_cross_attention_localization.py  # 結合が実際に再構築されたか
```

`classes_pos` / `classes_wrongpos` は案 A として不採用にしたので、
**学習済みモデルの評価からは外す**（`--variants` で選択可能にする）。
1 run あたり 654×2 回の推論を節約でき、4 通りの評価で計 1.5 時間ほど浮く。
学習前ベースラインの値は測定済みなので下表に残す。

指標: `abs_rel`, `delta1`, `BF1`, `recall ON edge`, `recall OFF edge`

**T-both は空間バイアス ON と OFF の両方で評価する。**

| 設定 | 意味 |
|------|------|
| バイアス ON | 実際に運用する構成 |
| バイアス OFF | **B が何を学習したか**を単離する。C の寄与を外した素の状態 |

`eval_cross_attention_localization.py` は**必ずバイアス OFF で**走らせる。
ON のまま lift を測るとバイアスが支配し、学習で結合が育ったかが見えない。

### Hypersim hold-out

NYUv2 は実写、学習は Hypersim（合成）なので、結果が出ても
「テキストが効かなかった」のか「ドメインが違って効かなかった」のか分離できない。
**学習に使わない Hypersim のシーンを 300 枚ほど取り分け、同じ指標で評価する。**

- シーン単位で分割する（`ai_XXX_YYY` 単位）。同一シーンの別フレームは照明違いに近く、
  フレーム単位で割ると hold-out にならない
- GT 深度は Hypersim に付属（`geometry_hdf5` の `depth_meters.hdf5`）
- 検出は `hypersim_yolo_detections` に既にある

**NYUv2 で効かず hold-out で効いた場合はドメインギャップが主因**であり、
手法自体は否定されない。逆に hold-out でも効かなければ、テキスト条件付けが
機能していないと判断できる。

学習前のベースライン（NYUv2 654 枚、比較の基準）:

| プロンプト | abs_rel | BF1 | recall ON | recall OFF |
|-----------|--------:|----:|----------:|-----------:|
| empty | 0.05000 | 0.0693 | 47.7% | 23.9% |
| classes | 0.05180 | 0.0736 | 52.9% | 27.3% |
| shuffled | 0.05176 | 0.0738 | 52.5% | 27.0% |
| generic | 0.05216 | 0.0746 | 53.8% | 27.8% |
| classes_pos | 0.05181 | 0.0742 | 53.5% | 27.6% |
| classes_wrongpos | 0.05182 | 0.0742 | 53.5% | 27.6% |

cross-attention lift のベースライン: **1.079**（完全な結合なら 5.18、空間無視なら 1.00）

## 5. 判定基準（着手前に固定する）

主要評価項目: **`recall OFF edge`**（今回特定した欠損そのもの）

成功と言うには **5 つすべて**が必要:

| # | 条件 | 何を排除するか |
|---|------|---------------|
| a | `classes` > `shuffled`（同一 run 内） | 摂動ではなく**意味**が効いている |
| b | T-text または T-both の `classes` > T-null の `classes` | LoRA ではなく**テキスト**が効いている |
| c1 | abs_rel が T-null/`empty` 以下 | テキストが**追加コストを生んでいない**（実験の妥当性） |
| c2 | abs_rel が **0.05000（学習前）以下** | **手法として使い物になる**（実用上の価値） |
| d | **バイアス OFF での** lift が 1.079 から明確に上昇 | **機構が実際に変化**した |

a〜c が通って **d が通らない場合は信用しない**。
結合が変わっていないのに指標だけ動いたなら、別の経路の副作用であり、
原因を特定するまで結論を出さない。

**c1 と c2 を分ける理由**: c1 だけだと抜け穴がある。
T-null/`empty` が 0.056（税を払った状態）、T-text/`classes` が 0.055 なら c1 は通るが、
公式 Lotus の 0.05000 より悪い。**c1 は実験が読めるかを見ており、
手法として使えるかは c2 でしか判定できない。**
c1 のみ通過なら「テキストは効いたが LoRA の税を埋められなかった」という結論になる。

### B と C の切り分け

| 観測 | 解釈 |
|------|------|
| T-text で (d) が通る | **B が働いた**。結合を学習で再構築できた |
| T-both のバイアス OFF で (d) が通る | C の存在下でも B が育った。両立する |
| T-both は ON で改善するが OFF で lift が動かない | **C 単独の効果**。位置を手渡さないと成立しない |
| T-both ≈ T-text ≈ T-null | どちらも効いていない。中断 |

これまで対照なしの数値が繰り返し覆っている
（[`object_oracle_control.md`](object_oracle_control.md): 上限 2.1% → 正味 −0.60%、
[`boundary_f1_findings.md`](boundary_f1_findings.md): abs_rel +1.5% → BF1 +313%）。
基準を先に固定するのはそのため。

## 6. 中断条件

| 時点 | 条件 | 対応 |
|------|------|------|
| checkpoint-1000 | **T-text / T-both の lift が T-null の lift と差がない**（バイアス OFF 測定） | 結合が再構築されていない。step 数か rank を上げるか、`text_encoder` の凍結解除を検討 |
| checkpoint-1500 | T-text ≈ T-both ≈ T-null（全指標） | テキストも位置も寄与していない。中断 |
| 任意 | abs_rel が empty ベースライン（0.05000）から 15% 以上劣化 | 税が過大。LoRA 対象か LR を見直す |

> lift の中断判定は**絶対値ではなく T-null との差**で見る。
> 学習そのものが lift をどう動かすかは未知なので、
> 「1.2 未満なら中断」のような固定閾値には根拠がない。
> T-null は同じ LoRA 容量・同じ step 数でテキストを使わずに学習した対照なので、
> 「テキストを与えたことによる結合の変化」はこの差にしか現れない。

---

## 7. 既知のリスク

1. **語彙が狭い。** Hypersim の検出は 27 クラスのみ（COCO の屋内部分集合）。
   意味情報の粒度が粗く、「chair」で表せる区別しか学べない。
2. **ドメインギャップ。** 学習は Hypersim（合成）、評価は NYUv2（実写）。
   Hypersim の hold-out でも評価し、ギャップの寄与を分離する。
3. **検出ゼロが 39.5%。** 実効的な学習信号は残り 60.5% から得る。
   step 数を 3000 に取ったのはこの希釈を見込んでのこと。
4. **先行研究。** VPD / ECoDepth / WorDepth が意味・テキスト条件付けの深度推定を扱う。
   結果が出た段階で、差別化点（Lotus の未使用テキスト経路、単一ステップ推論の維持）を
   明示できるか改めて確認する。
5. **名前に位置がない問題。** 学習で cross-attention が結合を獲得すれば解決するが、
   獲得しなければ残る。判定基準 (d) がこれを直接見ている。

---

## 8. 所要時間

| 工程 | 見積 |
|------|------|
| 2.1〜2.5 の実装 + hold-out 分割 | 1.5 日 |
| T-null 学習（3000 step） | 67 分 |
| T-text 学習（3000 step） | 67 分 |
| T-both 学習（3000 step） | 67 分 |
| 評価（NYUv2: T-null / T-text / T-both ON / T-both OFF の 4 通り、variant を 4 つに削減） | 約 1.5 時間 |
| 評価（Hypersim hold-out 300 枚、同 4 通り） | 約 40 分 |
| **1 サイクル合計** | **2 日** |

---

## 9. 実装順序

1. `utils/object_prompt.py` — 共通のプロンプト構築とトークン位置
2. `eval_text_prompt_conditioning.py` / `eval_cross_attention_localization.py` を
   それを使うよう修正（**学習前に**書式を固定する）
3. **同 2 本に `--lora_path` と `--class_token_spatial_bias` を追加**（2.5）。
   公式モデルに対して走らせ、**追加前と同じ数値が出ることを確認**してから先へ進む
4. `utils/object_spatial_attention.py` に `build_class_token_spatial_bias` を追加 + 単体テスト
   （`tests/test_object_spatial_attention.py` に既存のテストがあるので同じ形で）。
   テストで確認するのは 3 点: 形状 `[B,1,Q,77]` / **クラストークンの列以外がゼロ**である /
   同一クラス複数インスタンスで max が取られている
5. **Hypersim を hold-out 分割**（`ai_XXX_YYY` 単位、300 枚程度）。
   学習側の除外リストと評価スクリプトの両方で同じ分割を参照する
6. `train_lotus_d.py` の 2.1 / 2.2 / 2.4
7. `--max_train_steps=20` で煙試験。ログで確認するのは 3 点:
   - `LoRA blocks=text  trainable params=...`（0 でないこと）
   - サンプルごとに異なるプロンプトが実際に流れていること（数件を print）
   - バイアス ON のとき、クラストークンの列にのみ値が入っていること（形状と非ゼロ位置）
8. **T-null** を先に回す（対照が先にあれば、以降の結果をすぐ判定できる）
9. **T-text**（B 単独）
10. **T-both**（B + C）
11. 評価と判定。(d) は必ずバイアス OFF で測る

> 起動確認について: 過去に 2 回、学習が起動していないのに起動したと報告した事故がある
> （[`adaptation_placement_findings.md`](adaptation_placement_findings.md) 第 4 節）。
> **「意図した設定でループに入った」ログを確認するまで、開始とみなさない。**
