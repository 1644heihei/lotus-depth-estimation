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

---

## 1. 既に揃っているもの

| 資産 | 状態 |
|------|------|
| Hypersim の YOLO 検出 | `D:/lotus/data/hypersim_yolo_detections` に **59,544 件**、`class_name` 付き |
| 検出の読み出し | `utils/object_detection_cache.load_detections(path, root)` |
| 学習バッチ中の画像パス | `batch["image_pathes"]`（`train_lotus_d.py:1737` で既に使用） |
| LoRA の対象指定 | `--lora_target_blocks`（正規表現、マッチ 0 件で例外停止） |
| 評価スクリプト | `eval_text_prompt_conditioning.py`（6 プロンプト × 4 指標、対照込み） |
| 機構の検証 | `eval_cross_attention_localization.py`（lift + 2 対照） |

**Hypersim の検出統計**（400 件標本）: 平均 1.60 個/画像、**39.5% は検出ゼロ**、クラス 27 種。
検出ゼロの画像は自然に空プロンプトになるので、暗黙のプロンプトドロップアウトとして働く。

---

## 2. コード変更（3 箇所）

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
2. `padding="do_not_pad"` → **`padding="max_length"`**。
   サンプルごとに長さが変わるため、バッチ化には固定長が要る。
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

`utils/object_prompt.py` を新設し、`build_class_prompt(image_path, root, thr)` を置く。
`train_lotus_d.py` と `eval_text_prompt_conditioning.py` の両方がこれを呼ぶ。
**学習と評価で書式がずれる事故を構造的に防ぐ。**

---

## 3. 学習する 2 本

| run | LoRA 対象 | 学習時のプロンプト | 目的 |
|-----|----------|------------------|------|
| **T-text** | `text` | クラス名、dropout 0.1 | 本命 |
| **T-null** | `text` | 常に `""` | **fine-tune 税の分離**。容量・step 数を揃えた対照 |

共通設定（`run_lora_depth_placement.ps1` に準拠）:
rank 8 / alpha 16 / LR 1e-5 cosine / warmup 200 / batch 8 / res 512 /
**3000 step**（checkpoint 500 ごと）/ seed 42

3000 step ≈ **67 分**/run。2 本で約 2 時間 15 分。

> T-null を必ず回すこと。これまでの実験は例外なく fine-tune 税を払っており、
> T-text 単独では「テキストが効いた」のか「LoRA が効いた/害した」のか分離できない。

---

## 4. 評価

既存スクリプトを学習済み LoRA に向けて実行するだけ。

```
eval_text_prompt_conditioning.py   # empty / classes / shuffled / generic / classes_pos / classes_wrongpos
eval_cross_attention_localization.py  # 結合が実際に再構築されたか
```

指標: `abs_rel`, `delta1`, `BF1`, `recall ON edge`, `recall OFF edge`

学習前のベースライン（NYUv2 654 枚、比較の基準）:

| プロンプト | abs_rel | BF1 | recall ON | recall OFF |
|-----------|--------:|----:|----------:|-----------:|
| empty | 0.05000 | 0.0693 | 47.7% | 23.9% |
| classes | 0.05180 | 0.0736 | 52.9% | 27.3% |
| shuffled | 0.05176 | 0.0738 | 52.5% | 27.0% |
| generic | 0.05216 | 0.0746 | 53.8% | 27.8% |
| classes_pos | 0.05181 | 0.0742 | 53.5% | 27.6% |
| classes_wrongpos | 0.05182 | 0.0742 | 53.5% | 27.6% |

---

## 5. 判定基準（着手前に固定する）

主要評価項目: **`recall OFF edge`**（今回特定した欠損そのもの）

成功と言うには **4 つすべて**が必要:

| # | 条件 | 何を排除するか |
|---|------|---------------|
| a | T-text の `classes` > T-text の `shuffled` | 摂動ではなく**意味**が効いている |
| b | T-text の `classes` > T-null の `classes` | LoRA ではなく**テキスト**が効いている |
| c | T-text/`classes` の abs_rel が T-null/`empty` 以下 | **税を上回る**利得がある |
| d | cross-attention の lift が 1.079 から明確に上昇 | **機構が実際に変化**した |

a〜c が通って **d が通らない場合は信用しない**。
結合が変わっていないのに指標だけ動いたなら、別の経路の副作用であり、
原因を特定するまで結論を出さない。

これまで対照なしの数値が繰り返し覆っている
（[`object_oracle_control.md`](object_oracle_control.md): 上限 2.1% → 正味 −0.60%、
[`boundary_f1_findings.md`](boundary_f1_findings.md): abs_rel +1.5% → BF1 +313%）。
基準を先に固定するのはそのため。

---

## 6. 中断条件

| 時点 | 条件 | 対応 |
|------|------|------|
| checkpoint-1000 | lift が 1.2 未満 | 結合が再構築されていない。step 数か rank を上げるか、`text_encoder` の凍結解除を検討 |
| checkpoint-1500 | T-text ≈ T-null（全指標） | テキストが寄与していない。中断 |
| 任意 | abs_rel が empty ベースラインから 15% 以上劣化 | 税が過大。LoRA 対象か LR を見直す |

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
| 2.1〜2.3 の実装 | 半日 |
| T-text 学習（3000 step） | 67 分 |
| T-null 学習（3000 step） | 67 分 |
| 評価 2 本 × 2 run | 約 1 時間 |
| **1 サイクル合計** | **1 日** |

---

## 9. 実装順序

1. `utils/object_prompt.py` — 共通のプロンプト構築
2. `eval_text_prompt_conditioning.py` をそれを使うよう修正（**学習前に**書式を固定する）
3. `train_lotus_d.py` の 3 箇所
4. `--max_train_steps=20` で煙試験。ログで確認するのは 2 点:
   - `LoRA blocks=text  trainable params=...`（0 でないこと）
   - サンプルごとに異なるプロンプトが実際に流れていること（数件を print）
5. T-null を先に回す（対照が先にあれば、T-text の結果をすぐ判定できる）
6. T-text
7. 評価と判定

> 起動確認について: 過去に 2 回、学習が起動していないのに起動したと報告した事故がある
> （[`adaptation_placement_findings.md`](adaptation_placement_findings.md) 第 4 節）。
> **「意図した設定でループに入った」ログを確認するまで、開始とみなさない。**
