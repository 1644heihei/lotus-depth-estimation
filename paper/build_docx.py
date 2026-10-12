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


def convert(tex: Path, out: Path) -> None:
    src = tex.read_text(encoding="utf-8")
    title, author = title_of(src), author_of(src)

    # pandoc expands \newcommand itself, so \bfone needs no help; substituting it here
    # would rewrite the definition line as well and the file would stop parsing.
    body = src

    tmp = out.with_suffix(".pandoc.tex")
    tmp.write_text(body, encoding="utf-8")
    try:
        pypandoc.convert_file(
            str(tmp), "docx", outputfile=str(out),
            extra_args=[f"--resource-path={HERE}", "--standalone",
                        "--metadata", f"title={title}",
                        "--metadata", f"author={author}"])
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{out.name}: {out.stat().st_size // 1024} KB   title={title[:48]!r}")


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
