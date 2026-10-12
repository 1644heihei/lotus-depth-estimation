# Paper draft

Two versions of the same paper.

| file | language | for |
|---|---|---|
| `main.tex` | English | the submission |
| `main_ja.html` + `main_ja.pdf` | Japanese | reading, checking, the thesis |

The Japanese version is single-column A4 and runs 8 pages, which is about the
same content as 6 two-column pages. `./make_pdf.sh` regenerates the PDF with
headless Edge or Chrome, so no LaTeX or pandoc install is needed. Both versions
carry the same numbers; if one changes, change the other.


`main.tex` — ~6 pages, two-column, written against IEEEtran. Swap the
`\documentclass` line for the venue template when it arrives; nothing else in
the file depends on IEEEtran.

Every number in it comes from a run recorded in `docs/contour_selector_results.md`
and reproducible from this repository. `check_tex.py main.tex` reports any
`\cite` without a `\bibitem`, any `\ref` without a `\label`, and the remaining
TODO markers.

## What still needs a human

- Author, affiliation, contact.
- `\bibitem{marigoldv2}` — the primary base model is cited as a placeholder.
- `\bibitem{edgeguided}` and `\bibitem{nerfrefine}` — found by search, authors
  and venue not verified against the papers themselves.
- No direct comparison against Ramamonjisoa et al. The paper says so explicitly
  and says why; do not let the SAM ablation be read as one.

## Figures

- `figures/qualitative_a.png`, `figures/qualitative_b.png` — from
  `viz_sharpening.py`. Regenerate with:

      python viz_sharpening.py --dataset nyuv2 --model marigold_v2 \
        --mask_cache_dir D:/lotus/data/oracle_cache/sam_auto48_sel_raw \
        --max_abs_rel 0.030294 --n_panels 6 \
        --out_dir output/figures/sharpening_mv2

  The earlier command omitted --model, so the prediction was aligned in disparity
  space against a log-depth model and every per-frame number in the caption was
  wrong; --mask_cache_dir must be passed explicitly because the registry would
  otherwise replace it with the SAM candidates and the figure would show no
  selection. --max_abs_rel is the split's median AbsRel.

- The pipeline figure is TikZ inside `main.tex`; it needs no external file.

## Word copies

`python build_docx.py` writes `output/paper_en.docx` and `output/paper_ja.docx`
through pandoc (pip install pypandoc_binary). These are for reading and tracked
changes, not for submission: the submission is the PDF build.sh produces, and the
Word rendering is single-column with none of the IEEE layout.
