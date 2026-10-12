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

- `figures/pipeline.png` — from `make_pipeline_figure.py`. It was a PNG with no source until the fifth-version review asked for its wording to change; the README had described it as TikZ inside `main.tex`, which it never was.

## Word copies

`python build_docx.py` writes `output/paper_en.docx` and `output/paper_ja.docx`
through pandoc (pip install pypandoc_binary). These are for reading and tracked
changes, not for submission: the submission is the PDF build.sh produces.

The Word rendering approximates the IEEE layout rather than reproducing it: Letter
paper, the same margins, two columns with the title block and the wide figure
spanning, Times at 10pt. It runs longer than six pages, which is expected - Word
neither hyphenates nor sets the same way, and the page limit is the PDF's.

Two things happen to the LaTeX before pandoc sees it. `\cite` is expanded to the
numbers `thebibliography` would print, because with no .bib to resolve against
pandoc drops citations silently. Columns are then injected as Word sections,
which is the only place Word keeps them; `docx_layout.py` has that part.

To build against the venue's own Word template instead:

      python build_docx.py --only ja --suffix _template \
        --template <conference-template-a4.docx>

`docx_template.py` does that part. The template cannot be used as supplied: its
lengths are written in points where OOXML wants integer twips, and its last
section references a footer pandoc does not carry over, so the output is a file
Word refuses to open - a two-line document is enough to reproduce it. Both are
repaired in a copy first. The styles are then pointed at the template's own names,
the references get a heading LaTeX prints and pandoc does not, and the caption
numbering is rewritten from "Fig."/"TABLE" to 図/表 so it agrees with the
Japanese body.
