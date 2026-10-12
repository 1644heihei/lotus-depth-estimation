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

import re
import sys
from pathlib import Path

import pypandoc

from docx_layout import (reference_docx, span_figures, span_title,
                         wide_figure_indices)

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


def convert(tex: Path, out: Path) -> None:
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
    ref = reference_docx(pypandoc.get_pandoc_path(),
                         out.with_name("_reference_two_column.docx"))
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

    # The break goes on the last paragraph above the abstract, which is the author line.
    spanned = span_title(out)
    wide = span_figures(out, wide_figure_indices(src))
    print(f"{out.name}: {out.stat().st_size // 1024} KB   two columns, "
          f"title spanning: {'yes' if spanned else 'NO - abstract not found'}, "
          f"full-width figures: {wide}   title={title[:40]!r}")


def main() -> int:
    out_dir = HERE.parent / "output"
    out_dir.mkdir(exist_ok=True)
    for name, docx in (("main.tex", "paper_en.docx"), ("main_ja.tex", "paper_ja.docx")):
        tex = HERE / name
        if not tex.is_file():
            print(f"missing {tex}")
            return 1
        convert(tex, out_dir / docx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
