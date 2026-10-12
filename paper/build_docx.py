r"""Produce a Word copy of each paper, for reading and commenting.

The submission is the PDF that build.sh produces from LaTeX; this is a second rendering
of the same source for people who want to leave tracked changes. It is deliberately not
the IEEE two-column layout - the point is a readable single column with the tables,
figures and bibliography intact, not a camera-ready substitute.

Two things pandoc will not do on its own with an IEEEtran file: it drops \title and the
\IEEEauthorblock* author block, and it leaves \bfone unexpanded if the macro is defined
with \newcommand after \documentclass but used before \begin{document} is parsed. Both
are handled here by preprocessing a copy, so main.tex itself stays the LaTeX the venue
wants.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pypandoc

from docx_layout import (reference_docx, span_figures, span_title,
                         wide_figure_indices)
from docx_template import (apply_a4_sections, drop_paragraph, localise_numbering,
                           normalise, remap_styles, style_bibliography)

HERE = Path(__file__).resolve().parent


def title_of(src: str) -> str:
    m = re.search(r"\\title\{(.+?)\}\s*\n\n", src, re.S)
    if not m:
        return ""
    t = m.group(1)
    t = t.replace("\\\\", " ")                 # the line break inside the title
    # The Japanese title sets its subtitle smaller; the size command and its braces are
    # layout, not text, and would otherwise appear verbatim in the Word heading.
    t = re.sub(r"\{\\(?:large|Large|normalsize|small)\s*(.*?)\}", r"\1", t, flags=re.S)
    return re.sub(r"\s+", " ", t).strip()


def author_of(src: str) -> str:
    """The author block as plain text, TODO markers and all."""
    m = re.search(r"\\author\{(.+?)\}\}\s*\n", src, re.S)
    if not m:
        return ""
    a = m.group(1)
    a = re.sub(r"\\IEEEauthorblock[NA]\{", "", a)
    a = re.sub(r"\\textit\{(.+?)\}", r"\1", a)
    a = a.replace("\\\\", "\n").replace("}", "")
    lines = [ln.strip() for ln in a.splitlines()]
    return ", ".join(ln for ln in lines if ln)


def number_citations(src: str) -> str:
    """Replace \\cite{key,...} with the [n] numbers thebibliography would have given.

    Numbering follows \\bibitem order, which is what IEEEtran prints, so the Word copy
    and the PDF cite the same numbers. An unknown key is left visible as [?key] rather
    than dropped, because a silently missing citation is what this is fixing.
    """
    order = {m.group(1): i + 1 for i, m
             in enumerate(re.finditer(r"\\bibitem\{([^}]+)\}", src))}

    def one(m: re.Match) -> str:
        keys = [k.strip() for k in m.group(1).split(",")]
        return ", ".join(f"[{order[k]}]" if k in order else f"[?{k}]" for k in keys)

    return re.sub(r"\\cite\{([^}]+)\}", one, src)


def convert(tex: Path, out: Path, template: Path | None = None) -> None:
    """Render `tex` to `out`. With `template`, fit it to the venue's Word template
    instead of the generic two-column layout; see docx_template.py for what that takes
    beyond handing pandoc the file."""
    src = tex.read_text(encoding="utf-8")
    title, author = title_of(src), author_of(src)

    # pandoc expands \newcommand itself, so \bfone needs no help; substituting it here
    # would rewrite the definition line as well and the file would stop parsing.
    # \cite is a different matter: with a hand-written thebibliography and no .bib there
    # is nothing for pandoc to resolve against, and it drops the citation silently -
    # "Boundary-specific metrics do ." Numbering them here keeps the references readable.
    body = number_citations(src)

    tmp = out.with_suffix(".pandoc.tex")
    tmp.write_text(body, encoding="utf-8")
    if template:
        # Never the template as supplied: its lengths are in points where the schema
        # wants integer twips, and its last section points at a footer pandoc does not
        # carry over. See docx_template.normalise.
        ref, fixed = normalise(template, out.with_name("_template_normalised.docx"))
    else:
        ref, fixed = reference_docx(pypandoc.get_pandoc_path(),
                                    out.with_name("_reference_two_column.docx")), 0
    try:
        pypandoc.convert_file(
            str(tmp), "docx", outputfile=str(out),
            extra_args=[f"--resource-path={HERE}", "--standalone",
                        f"--reference-doc={ref}",
                        "--metadata", f"title={title}",
                        "--metadata", f"author={author}"])
    finally:
        tmp.unlink(missing_ok=True)
        ref.unlink(missing_ok=True)

    wide_figs = wide_figure_indices(src)
    if template:
        moved = remap_styles(out)
        n_items = len(re.findall(r"\\bibitem\{", src))
        ja = "_ja" in tex.stem
        width = re.search(r"\\begin\{thebibliography\}\{([^}]*)\}", src)
        if width and width.group(1):
            drop_paragraph(out, width.group(1))
        if ja:
            # The template numbers captions "Fig. %1." and "TABLE %1." in Roman; the
            # Japanese body says 図 and 表, and pandoc writes its cross-references in
            # arabic, so both the word and the format have to follow.
            localise_numbering(out, {"2": ("図 %1.", "decimal"),
                                     "9": ("表 %1. ", "decimal")})
        style_bibliography(out, n_items, "参考文献" if ja else "References")
        spanned, wide = apply_a4_sections(out, wide_figs)
        print(f"{out.name}: {out.stat().st_size // 1024} KB   template layout, "
              f"{fixed} lengths normalised, {moved} styles remapped, {n_items} refs, "
              f"title spanning: {'yes' if spanned else 'NO'}, "
              f"full-width figures: {wide}")
        return

    # The break goes on the last paragraph above the abstract, which is the author line.
    spanned = span_title(out)
    wide = span_figures(out, wide_figs)
    print(f"{out.name}: {out.stat().st_size // 1024} KB   two columns, "
          f"title spanning: {'yes' if spanned else 'NO - abstract not found'}, "
          f"full-width figures: {wide}   title={title[:40]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--template", type=Path, default=None,
                    help="The venue's Word template. Its styles and layout are used "
                         "instead of the generic two-column rendering.")
    ap.add_argument("--only", choices=("en", "ja"), default=None)
    ap.add_argument("--suffix", default="",
                    help="Appended to the output names, to keep a template build "
                         "beside the plain one.")
    args = ap.parse_args()
    if args.template and not args.template.is_file():
        print(f"missing template {args.template}")
        return 1

    out_dir = HERE.parent / "output"
    out_dir.mkdir(exist_ok=True)
    jobs = [("main.tex", "paper_en"), ("main_ja.tex", "paper_ja")]
    if args.only:
        jobs = [j for j in jobs if j[1].endswith(args.only)]
    for name, stem in jobs:
        tex = HERE / name
        if not tex.is_file():
            print(f"missing {tex}")
            return 1
        convert(tex, out_dir / f"{stem}{args.suffix}.docx", args.template)
    return 0


if __name__ == "__main__":
    sys.exit(main())
