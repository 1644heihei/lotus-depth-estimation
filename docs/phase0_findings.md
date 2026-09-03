# Phase 0 実施記録 — Oracle 上限測定と解像度スイープ

実施: 2026-08-27
ブランチ: `feature/object-depth-aux-loss`
計画: [`lotus_improvement_plan.md`](lotus_improvement_plan.md)

物体検出ベースの手法がすべて失敗した後、「そもそも改善余地がどこにあるか」を
手法を作る前に測定した記録。所要時間は合計 1 時間弱。

---

## 0. 要約

| 発見 | 数値 | 帰結 |
|------|------|------|
| **全実験が非標準の解像度 512 で走っていた** | 公式デフォルトは **768** | **設定ミス。過去の全記録は公式に不利な条件だった** |
| 設定を公式既定に戻すだけで 9.8% 改善 | 0.05543 → **0.05000**（学習なし） | **以後のベースラインは 0.05000** |
| これまでの全実験の最良値 = 公式 Lotus と同等 | 0.05538 vs 0.05543（差 0.09%） | 数ヶ月分の実験で有意な改善ゼロ |
| **完璧な regressor でも改善は 2〜3% が上限** | Oracle A = **2.1% / 3.1%**（768） | **regressor 路線を破棄** |
| 物体領域の余地の 9 割は形状・境界 | Oracle B 19.2% vs A 2.1%（**9.1 倍**） | 深度値の注入ではなく解像度・境界を直すべき |
| **誤差は画像全体にほぼ均等分布** | 誤差密度 0.89〜1.03 | 物体は誤差の集中点ではない。全手法失敗の最終理由 |
| 1024 では −14.4% と悪化 | ピークは 768 の単峰形 | 素朴な解像度上げには上限 → タイル分割が必要 |
| **Oracle の結論は解像度に対して頑健** | 512 と 768 で gain% がほぼ一致 | 判定 1〜3 は設定ミスの影響を受けない |

---

## 1. 公式 Lotus ベースライン（新規計測）

これまで**全実験の基準となる「公式 Lotus を同一プロトコルで評価した値」が存在しなかった**。
過去に存在したのは 522枚 subset（`min_detections>=1`）での値のみで、
bbox-split 実験群（654枚）と母集団が異なり直接比較できなかった。

```powershell
python eval_regressor_predepth_nyuv2.py `
  --detail_model=jingheya/lotus-depth-d-v2-0-disparity `
  --regressor_dir=output/object_depth_regressor_v5_roi `
  --detail_artifacts_dir=D:/lotus/data/nyuv2_detail_artifacts/test `
  --condition_mode=unconditioned --detection_score_thr=0.5 `
  --processing_res=512 --seed=42 --half_precision `
  --output_dir=output/eval_baseline_official_654
```

結果: **abs_rel 0.05543 / δ1 0.9698 / roi 0.04475 / bg 0.05226**（654枚）

これと過去の全実験を並べると:

| モデル | abs_rel | roi_abs_rel | δ1 |
|--------|---------|-------------|-----|
| **公式 Lotus（無改造）** | **0.05543** | **0.04475** | **0.9698** |
| v5 step-250（全実験中の最良） | 0.05538 | 0.04476 | 0.9697 |
| v6 ablation step-250 | 0.05574 | 0.04526 | 0.9691 |
| v2 step-3000 | 0.07371 | 0.05387 | 0.9320 |

最良値との差は **0.00005 = 相対 0.09%**、すなわち誤差。
**「ベース重みをほとんど動かさなかった状態」が最良だった。**

---

## 2. Oracle 上限測定

### 2.1 実装

`eval_object_oracle_ceiling.py`（新規）

公式 Lotus の予測を後処理し、物体領域を GT で置き換えて
「完璧な物体情報があったら abs_rel はどこまで下がるか」を測る。学習も再推論も不要。

```powershell
# 主測定（公式デフォルト解像度）
python eval_object_oracle_ceiling.py --processing_res=768 --half_precision `
  --output_dir=output/eval_object_oracle_ceiling_res768

# 参考（過去の全実験と同じ 512）
python eval_object_oracle_ceiling.py --processing_res=512 --half_precision `
  --output_dir=output/eval_object_oracle_ceiling
```

| variant | 置換内容 | 測っているもの |
|---------|----------|----------------|
| `baseline` | なし | 公式 Lotus そのもの |
| `A_bbox` / `A_mask` | 物体ごとに**深度レベルのみ** GT に合わせる（形状は保持）<br>`depth[Mi] *= median(gt[Mi]) / median(depth[Mi])` | **完璧な regressor** の上限<br>（regressor は物体 1 個につき深度 1 個しか出せない） |
| `B_bbox` / `B_mask` | 物体領域を **GT で完全置換** | 物体領域を扱う**あらゆる手法**の上限 |
| `BG` | 逆に**物体以外**を GT で完全置換 | 誤差のうち背景側の取り分 |

実装上の要点:

- **alignment は置換の前**に行う（`align_to_gt()`）。置換後だと GT が least-square fit に
  漏れて循環参照になる
- 置換は **GT が有効な画素のみ**（`gt` の穴を貼らない）
- Oracle A は**乗算補正**（abs_rel が相対誤差のため）
- seg マスクは現行キャッシュに保存されていない（[原因4](lotus_improvement_plan.md)）ため、
  NYUv2 test 654枚に yolov8n-seg を流して別途キャッシュを作成
- Lotus 予測（`.npy` float16）と seg マスク（`.npz` packed bits）を
  `D:/lotus/data/oracle_cache/` にキャッシュし、6 variant で使い回す

**実装時のハマりどころ**: fp16 パイプラインに float32 テンソルを渡すと
`RuntimeError: Input type (float) and bias type (struct c10::Half) should be the same` になる。
既存 `eval_regressor_predepth_nyuv2.py` と同様に `torch.autocast(device_type=...)` で囲む必要がある。

### 2.2 計算の検証

2 つの独立した整合性チェックを通過している:

1. `baseline` = **0.05543** — §1 の独立計測と完全一致
2. `B_bbox` の gain（0.01115）＋ `BG` の gain（0.04428）= **0.05543** = baseline
   → 物体領域と背景が誤差を過不足なく分割している

### 2.3 結果（NYUv2 654枚）

**主表 — `processing_res=768`（公式デフォルト）** / `output/eval_object_oracle_ceiling_res768/`

| variant | abs_rel | δ1 | gain | gain% | 対象面積 |
|---------|---------|-----|------|-------|----------|
| baseline（公式 Lotus） | 0.05000 | 0.9711 | — | — | — |
| **Oracle A bbox** | 0.04892 | 0.9713 | 0.00108 | **2.1%** | 21.3% |
| **Oracle A mask** | 0.04843 | 0.9719 | 0.00157 | **3.1%** | 14.9% |
| Oracle B bbox | 0.04041 | 0.9752 | 0.00959 | 19.2% | 21.3% |
| Oracle B mask | 0.04338 | 0.9738 | 0.00662 | 13.2% | 14.9% |
| **Oracle BG** | 0.00958 | 0.9963 | 0.04042 | **80.8%** | 78.7% |

（654枚中 132枚は検出物体ゼロ）

**参考 — `processing_res=512`（過去の全実験と同条件）** / `output/eval_object_oracle_ceiling/`

| variant | 512 abs_rel | 512 gain% | 768 gain% |
|---------|-------------|-----------|-----------|
| baseline | 0.05543 | — | — |
| Oracle A bbox | 0.05430 | 2.0% | 2.1% |
| Oracle A mask | 0.05372 | 3.1% | 3.1% |
| Oracle B bbox | 0.04428 | 20.1% | 19.2% |
| Oracle B mask | 0.04790 | 13.6% | 13.2% |
| Oracle BG | 0.01115 | 79.9% | 80.8% |

> [!NOTE]
> **gain% は 2 つの解像度でほぼ一致しており、判定 1〜3 は解像度設定に対して頑健である。**
> §3.3 で判明した設定ミスは、Oracle が示す結論そのものには影響しない。
> なお 768 では物体領域の誤差密度がわずかに下がっており（下表）、
> 正しい解像度では物体はさらに「攻めどころではなくなる」方向に動く。

### 2.4 判定 1 — regressor 路線は破棄

**Oracle A = 2.1%（bbox）/ 3.1%（mask）で、測定前に設定した 5% の閾値を下回った**
（512 でも 2.0% / 3.1% と同じ）。

物体ごとの深度を完全に言い当てる regressor があっても改善は 2〜3% が上限。
現行 regressor の実測精度は NYUv2 で abs_rel=0.374 であり、上限の数分の一しか達成できない。
**regressor による物体深度の注入は原理的に投資に見合わない。**

### 2.5 判定 2 — 余地は「深度レベル」ではなく「形状」

Oracle B bbox（19.2%）は Oracle A bbox（2.1%）の **9.1 倍**。
物体領域の改善余地 19 ポイントのうち **17 ポイントは形状・境界**由来で、
深度レベル（距離）由来は 2 ポイントしかない。

### 2.6 判定 3 — 誤差は画像全体にほぼ均等分布（最重要）

gain% を対象面積で割った「単位面積あたりの誤差密度」（`processing_res=768`）:

| 領域 | 面積 | gain% | 誤差密度 | 512 での密度 |
|------|------|-------|----------|--------------|
| seg マスク内（物体そのもの） | 14.9% | 13.2% | **0.89** | 0.91 |
| bbox 内 | 21.3% | 19.2% | **0.90** | 0.94 |
| bbox 内のマスク外（矩形の余白） | 6.4% | 6.0% | **0.94** | 1.02 |
| 背景（bbox 外） | 78.7% | 80.8% | **1.03** | 1.02 |

正しい解像度（768）では物体領域の誤差密度がさらに下がり（0.89〜0.90）、
背景（1.03）との差が開いている。**物体は「攻めどころ」から一層遠ざかる。**

**誤差密度はどの領域でもほぼ 1.0。物体領域が特別に悪いわけではない。**
物体は「誤差が集中した攻めどころ」ではなく、単に画素の 15〜21% を占め、
それに見合った 14〜20% の誤差を持っているだけだった。
物体領域を完璧にしても**面積分の誤差しか回収できない** —
これがあらゆる物体条件付け手法が失敗した最終的な理由である。

副次的な確認: bbox 内のマスク外領域（密度 1.02）は背景（1.02）と同じ挙動を示す。
**矩形の余白は「物体の一部」ではなく背景そのもの**であり、
そこに物体深度を貼る現行実装が誤りであることが定量的に裏付けられた。

---

## 3. 解像度スイープ

### 3.1 動機

判定 2（余地は形状・境界）と、Lotus が VAE で 8 倍ダウンサンプルする構造
（`processing_res=512` → latent 64×64）から、
「注入しようとしていたディテールは、実は潰れていた解像度ではないか」という仮説を立て、
`--processing_res` のみを変えて検証した。公式 Lotus・学習なし・他条件はすべて同一。

```powershell
# $R = 640, 768, 896, 1024
python eval_regressor_predepth_nyuv2.py `
  --detail_model=jingheya/lotus-depth-d-v2-0-disparity `
  --regressor_dir=output/object_depth_regressor_v5_roi `
  --detail_artifacts_dir=D:/lotus/data/nyuv2_detail_artifacts/test `
  --condition_mode=unconditioned --detection_score_thr=0.5 `
  --processing_res=$R --seed=42 --half_precision `
  --output_dir=output/eval_baseline_official_654_res$R
```

### 3.2 結果（NYUv2 654枚・公式 Lotus・学習なし）

| processing_res | latent | abs_rel | δ1 | roi_abs_rel | bg_abs_rel | vs 512 |
|----------------|--------|---------|-----|-------------|------------|--------|
| 512（従来の全実験） | 64² | 0.05543 | 0.9698 | 0.04475 | 0.05226 | — |
| 640 | 80² | 0.05123 | **0.9723** | 0.04069 | 0.04809 | **+7.6%** |
| **768** | 96² | **0.05000** | 0.9711 | **0.03818** | **0.04693** | **+9.8%** |
| 896 | 112² | 0.05574 | 0.9670 | 0.04144 | 0.05233 | −0.5% |
| 1024 | 128² | 0.06342 | 0.9599 | 0.04637 | 0.05964 | −14.4% |

768 をピークとする明確な単峰形。ノイズではない。

### 3.3 発見 1 — これは「新発見」ではなく **設定ミスの修正** だった

> [!IMPORTANT]
> **公式 Lotus のデフォルト `processing_res` は 768 である。512 ではない。**
>
> - `pipeline.py:1041` — `LotusDPipeline.default_processing_resolution = 768`
> - `eval.py:78` — `help="Maximum resolution of processing. ... Default: 768."`
> - `datasets/eval/depth/configs/data_eth3d.yaml` — `processing_res: 768`
>
> つまり **この研究の全実験（v1〜v6・object attention・pre-depth fusion）は、
> 公式より低い非標準の解像度 512 で走っていた。**
> 「+9.8% の改善」は新しいレバーの発見ではなく、**公式の既定値に戻しただけ**である。

これは 2 つの意味で重要:

1. **「公式 Lotus ベースライン 0.05543」自体が、公式設定より劣化した値だった。**
   真の公式ベースラインは **0.05000（res=768）**。
2. **過去の物体条件付け実験は、ハンデを負わせたベースラインと比較してなお負けていた。**
   実際の差は記録より大きい。

数ヶ月分の物体条件付け実験（LoRA 学習・attention・bbox loss・regressor）が
公式 Lotus を 1 度も上回れなかったのに対し、
**設定を公式デフォルトに戻すだけで abs_rel が 9.8% 改善した。学習は一切していない。**

さらに **roi_abs_rel の改善（−14.7%）が全体（−9.8%）より大きい**。
物体条件付けが狙っていた「物体領域のディテール改善」は、
**正しい解像度で推論するだけで、しかもより大きく得られていた**。

**物体条件付けで注入しようとしていた「ディテール」の正体は、単に潰れていた解像度だった。**

### 3.3.1 なぜ 512 になっていたのか（原因の特定）

**データの問題ではない。** ローカルの NYUv2 は 640×480 で、これは NYUv2 labeled の
ネイティブ解像度そのもの。低解像度で保存されていたわけではない。

原因は **設定の継承ミス（カテゴリエラー）** で、以下の経路で混入した:

```
train_object_depth_regressor.py:255
    config["processing_res"] = 512      ← regressor の「ROI 特徴量抽出時のリサイズ解像度」
                ↓ config.json に保存
eval_regressor_predepth_nyuv2.py:378-382
    processing_res = args.processing_res if not None
                     else regressor.config.get("processing_res")   ← ここで流用
                ↓ 拡散パイプラインの推論解像度として使われる
train_scripts/eval_*.ps1
    --processing_res=512                ← 以降、全スクリプトに明示的にコピペ伝播
```

**regressor の 512 は「ROI 画像パッチを切り出すときのリサイズ解像度」**であって、
**Lotus 拡散パイプラインの推論解像度とはまったく別物**である。
`eval_regressor_predepth_nyuv2.py` がこれを流用したことが根本原因。

512 という値自体は、Hypersim の学習解像度（`train_lotus_d.py --resolution_hypersim=512`）や
Stable Diffusion の標準解像度に合わせたものと思われる。
「学習が 512 なら推論も 512」という一見自然な推論だが、
**Lotus は VAE + UNet の全 convolutional 構成で解像度可変であり、
学習解像度と推論解像度を一致させる必要はない**（公式が 768 を既定値にしているのがその証拠）。

伝播先（すべて 512 で固定されていた）:

| ファイル | 箇所 |
|----------|------|
| `train_scripts/eval_bbox_split_nyuv2_v1〜v6.ps1` | `--processing_res=512` |
| `train_scripts/eval_object_attention_nyuv2.ps1` | 同 |
| `train_scripts/eval_three_conditions_nyuv2.ps1` / `_scannet.ps1` | 同 |
| `train_scripts/build_scannet_crop_predepth.ps1` | 同 |
| `scripts/visualize_per_sample_cases.py` | `default=512` |
| `eval_object_oracle_ceiling.py`（本 Phase で新規作成） | `default=512` → **768 に修正済み** |

### 3.3.2 影響範囲は「全実験」ではなく「単体評価スクリプトのみ」だった

追加調査の結果、**学習ループ内の validation は 768 で動いていた**ことが判明した。
`train_lotus_d.py::run_evaluation` の `gen_depth` は `pipe()` に `processing_res` を
渡しておらず、パイプライン既定（768）が使われる。

| 経路 | processing_res | 該当する記録 |
|------|----------------|--------------|
| `train_lotus_d.py::run_evaluation` → `pipe()` 未指定 | **768**（既定） | 学習ログの validation、`detail_training_runs.md` の mid-eval、<br>`object_attention_*.md` の validation 表 |
| `eval_regressor_predepth_nyuv2.py`（ps1 が 512 を明示） | **512**（設定ミス） | `object_bbox_loss_investigation.md` の全表、<br>`object_attention_*.md` の「本番 eval」表 |

**これは過去の解釈に交絡をもたらしていた。** 例えば
`object_attention_training_results.md` は「12k validation（654枚）と 8k 本番 eval（522枚）は
**評価母集団が異なる**ため直接比較しないこと」と注意していたが、
実際には **母集団だけでなく推論解像度（768 vs 512）も違っていた**。
交絡要因は 1 つではなく 2 つあった。

該当ドキュメントには 2026-08-27 付で注記を追加済み。

### 3.3.3 修正内容（2026-08-27）

| 対象 | 修正 |
|------|------|
| `eval_regressor_predepth_nyuv2.py:378-387` | フォールバック先を regressor config → **`detail_pipe.default_processing_resolution`（768）** に変更 |
| 同 summary.json | **`processing_res`（解決後の値）/ `seed` / `half_precision` を記録するよう追加**。従来はキー自体が無く、どの解像度で得た数値か追跡できなかった |
| `train_scripts/*.ps1`（10 本） | `--processing_res=512` → `$PROCESSING_RES = 768` に変数化。冒頭に経緯を注記 |
| `scripts/visualize_per_sample_cases.py` | `default=512` → `768` |
| `train_object_depth_regressor.py:255` | 値は **512 のまま**（regressor 自身の ROI 抽出解像度なので正しい）。<br>`roi_processing_res` という明示的なキーを追加し、拡散パイプラインとは無関係である旨を明記 |
| `eval_object_oracle_ceiling.py` | `default=512` → `768`。**予測キャッシュを解像度別ディレクトリに分離**（768 実行時に 512 のキャッシュを誤読するバグを修正） |

> [!WARNING]
> **`eval_regressor_predepth_nyuv2.py` のフォールバックが設計上の欠陥だった。**
> regressor の config が拡散パイプラインの推論解像度を決めてよい理由はない。
> **他モジュールの config を別モジュールの設定に流用しないこと。**
>
> 併せて、**実験の summary に「実際に使われた設定」を記録していなかったこと**が、
> このミスを数ヶ月間検出できなかった構造的な原因である。
> 既定値に依存した実行は、記録がなければ後から追跡できない。

### 3.4 発見 2 — 768 で頭打ち、以降は急落

896 でベースライン同等、1024 で −14.4% と大きく悪化。
モデルの学習解像度から離れすぎて **OOD（分布外）** になるためと考えられる。

**素朴な解像度上げには上限がある。** これを超えるには
モデルが得意な解像度（512〜768）のタイルに分割して推論・合成する必要がある
→ [`tiled_inference_plan.md`](tiled_inference_plan.md)

---

## 4. 注意点・限界

### 4.1 テストセット調整の懸念 — §3.3 の判明により解消

当初「`processing_res=768` を test の結果を見て選ぶのはテストセット調整ではないか」を
懸念事項として挙げていたが、**768 は公式 Lotus のデフォルト値**であることが判明したため、
この懸念は解消された。test 結果を見て選んだのではなく、**公式の既定設定に戻しただけ**である。

ただし以下は引き続き注意:

- **「768 が最良」と示すスイープ自体は test 上で行っている。**
  論文で「768 が最適」と主張するなら validation で示すこと。
  「公式デフォルトを使った」と述べるだけなら問題ない。
- **過去の全実験の数値は 512 で取得されている。**
  実験群どうしの内部比較の公平性は保たれているが、
  **公式 Lotus との比較としては過去の記録すべてが公式に不利な条件だった**（§3.3）。
  再掲・引用時は必ずこの点を注記すること。

### 4.2 roi / bg の比較は厳密には等価でない

roi_abs_rel と bg_abs_rel はそれぞれ独立に least-square alignment を行った値のため、
領域サイズと深度レンジの違いにより等価比較ではない（物体は小さく深度レンジが狭いため
有利に出やすい）。ただし「物体領域が明確な弱点である」という主張を支持しない点は変わらない。

### 4.3 NYUv2 は 640×480 と低解像度

`processing_res=768` は既に原画像を拡大している。つまり 512→768 の利得は
「原画像により多くの情報があったから」ではなく **「latent の容量が増えたから」**（64²→96²）。
したがってレバーは *ソース解像度* ではなく *latent 容量* である。
タイル分割でも同じレバーを引けるが、原画像が持つ情報量が上限を規定するため、
NYUv2 は手法の実力を示す題材としては不利。高解像度データセットでの追試が望ましい。

---

## 5. 成果物・パス

| パス | 内容 |
|------|------|
| `eval_object_oracle_ceiling.py` | **新規** Oracle 上限測定スクリプト |
| `output/eval_object_oracle_ceiling_res768/` | **Oracle 主結果**（公式デフォルト解像度） |
| `output/eval_object_oracle_ceiling/` | Oracle 参考結果（res 512、過去実験と同条件） |
| `output/eval_baseline_official_654/` | 公式 Lotus ベースライン（res 512） |
| `output/eval_baseline_official_654_res{640,768,896,1024}/` | 解像度スイープ |
| `D:/lotus/data/oracle_cache/lotus_pred/res{512,768}/` | Lotus 予測キャッシュ（.npy float16、解像度別） |
| `D:/lotus/data/oracle_cache/yolo_seg/` | YOLO-seg マスクキャッシュ（.npz packed bits、解像度非依存） |

---

## 6. 得られた教訓

1. **手法を作る前に上限を測る。** Oracle 測定と解像度スイープは合計 1 時間弱で完了し、
   数ヶ月分の実験が触れられなかった 9.8% の改善を明らかにした。
2. **ベースラインを必ず同一プロトコルで持つ。** 基準値がなかったため、
   「最良の結果」が実は「何もしないのと同じ」だったことに長期間気づけなかった。
3. **対照実験を最初から回す。** v6 アブレーション（bbox_loss OFF）を最初に走らせていれば、
   bbox_loss に効果がないことを数ヶ月早く判定できた。
4. **単純な設定を先に試す。** 複雑な conditioning 機構より前に、
   `processing_res` のような基本的な推論設定をスイープすべきだった。
5. **既定値を上書きするときは根拠を確認する。**（§3.3.1）
   `processing_res=512` は regressor の ROI 抽出解像度が拡散パイプラインに
   流用されたもので、公式既定 768 を無自覚に下回っていた。
   **他モジュールの config を別モジュールの設定に流用しない。**
6. **自分が新規に書いたコードにも同じ罠が入る。**
   本 Phase で作成した `eval_object_oracle_ceiling.py` も、当初 `default=512` を
   踏襲していた（修正済み）。既存コードからのコピーは前提ごと引き継ぐ。
