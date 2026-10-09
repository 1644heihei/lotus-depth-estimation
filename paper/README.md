# Paper draft

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

      python viz_sharpening.py --n_panels 6 --out_dir output/figures/sharpening_mv2

- The pipeline figure is TikZ inside `main.tex`; it needs no external file.
