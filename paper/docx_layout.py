r"""Give the Word copies the PDF's two-column layout.

Word stores columns in a section, not a style, so pandoc cannot be asked for them
directly. Two pieces are needed:

  reference_docx()  a reference document whose trailing section is two columns on US
                    Letter with the venue's margins, which pandoc copies into whatever
                    it produces.
  span_title()      a section break after the author block, so the title and authors run
                    the full width as they do in the PDF, and everything after them is
                    the two-column section. In OOXML a section break lives in the pPr of
                    the LAST paragraph of the section it ends, which is why this edits
                    the author paragraph rather than inserting anything of its own.

Measurements are twips, 1/1440 inch: Letter is 12240 by 15840, the margins and the
0.25 inch gutter are IEEEtran's.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

# A section break starts a new page unless it says otherwise, which would put the title
# alone on page one. Continuous keeps it on the same page and only changes the columns.
CONTINUOUS = '<w:type w:val="continuous"/>'
PAGE = ('<w:pgSz w:w="12240" w:h="15840"/>'
        '<w:pgMar w:top="1080" w:right="900" w:bottom="1440" w:left="900"'
        ' w:header="720" w:footer="720" w:gutter="0"/>')
TWO_COL = '<w:cols w:num="2" w:space="360" w:equalWidth="1"/>'
# Times for the Latin runs, a Japanese face for the CJK ones; Word picks per script.
BODY_FONT = ('<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
             'w:cs="Times New Roman" w:eastAsia="Yu Mincho"/>')
ONE_COL = '<w:cols w:num="1" w:space="360"/>'


def _rewrite_zip(src: Path, dst: Path, changes: dict[str, bytes]) -> None:
    """Copy a zip, replacing the named members. Rewritten rather than edited in place
    because a docx is a zip and Python cannot replace a member of one."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(
            dst, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = changes.get(item.filename, zin.read(item.filename))
            zout.writestr(item, data)


def reference_docx(pandoc: str, out: Path) -> Path:
    """pandoc's own reference document, with a two-column trailing section."""
    raw = out.with_suffix(".raw.docx")
    raw.write_bytes(subprocess.run(
        [pandoc, "--print-default-data-file", "reference.docx"],
        capture_output=True, check=True).stdout)

    with zipfile.ZipFile(raw) as z:
        doc = z.read("word/document.xml").decode("utf-8")
        styles = z.read("word/styles.xml").decode("utf-8")
    assert doc.count("<w:sectPr>") == 1, "expected one section in the reference document"
    doc = doc.replace("<w:sectPr>", "<w:sectPr>" + PAGE + TWO_COL, 1)

    # The PDF is Times at 10pt; pandoc's reference document is not. Setting it on
    # docDefaults rather than on each style is enough, since none of the styles this
    # document uses override the family.
    old = re.search(r"<w:rFonts[^/]*/>", styles)
    if old:
        styles = styles.replace(old.group(0), BODY_FONT, 1)
    styles = re.sub(r'<w:sz w:val="\d+"\s*/>', '<w:sz w:val="20"/>', styles, count=1)

    _rewrite_zip(raw, out, {"word/document.xml": doc.encode("utf-8"),
                            "word/styles.xml": styles.encode("utf-8")})
    raw.unlink()
    return out


def _text(paragraph: str) -> str:
    """The visible text of a <w:p>, joined across its runs.

    Word splits a line into runs wherever formatting changes, so a marker that reads as
    one phrase in the document can be several <w:t> elements and is not findable in the
    raw XML.
    """
    return "".join(re.findall(r"<w:t[^>]*>([^<]*)</w:t>", paragraph))


def wide_figure_indices(src: str) -> list[int]:
    """Which figures, in document order, are full width.

    LaTeX says this with figure* against figure; the docx cannot, because pandoc renames
    every image to its relationship id and the original file name is gone. Order is all
    that survives, so order is what this matches on.
    """
    out = []
    for i, m in enumerate(re.finditer(r"\\begin\{figure(\*?)\}", src)):
        if m.group(1):
            out.append(i)
    return out


def span_figures(docx: Path, which: list[int]) -> int:
    """Let the listed images run the full page width inside a two-column document.

    A section ends at the paragraph carrying its sectPr, so a full-width figure needs
    two: the paragraph before it ends the two-column run, and the figure's own paragraph
    ends a one-column section of exactly itself.
    """
    if not which:
        return 0
    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf-8")

    paras = list(re.finditer(r"<w:p\b[^>]*>.*?</w:p>", doc, re.S))
    images = [i for i, m in enumerate(paras) if "<a:blip" in m.group(0)]
    edits: dict[int, str] = {}
    for n in which:
        if n >= len(images):
            continue
        fig = images[n]
        if fig == 0:
            continue
        edits[fig - 1] = TWO_COL
        edits[fig] = ONE_COL
    if not edits:
        return 0

    out, pos = [], 0
    for i, m in sorted((i, paras[i]) for i in edits):
        out.append(doc[pos:m.start()])
        out.append(_with_section(m.group(0), edits[i]))
        pos = m.end()
    out.append(doc[pos:])

    tmp = docx.with_suffix(".tmp.docx")
    _rewrite_zip(docx, tmp, {"word/document.xml": "".join(out).encode("utf-8")})
    shutil.move(tmp, docx)
    return len(edits) // 2


def _with_section(para: str, cols: str) -> str:
    """Attach a section break to a paragraph, respecting CT_PPr's element order."""
    sect = f"<w:sectPr>{CONTINUOUS}{PAGE}{cols}</w:sectPr>"
    if "</w:pPr>" in para:
        return para.replace("</w:pPr>", sect + "</w:pPr>", 1)
    return re.sub(r"(<w:p\b[^>]*>)", r"\1" + f"<w:pPr>{sect}</w:pPr>", para, count=1)


def span_title(docx: Path, heading: str = "Abstract") -> int:
    """Make everything above `heading` span both columns.

    Located structurally rather than by the author's name: the break belongs on the last
    paragraph before the abstract, whatever that paragraph happens to say, which keeps
    working once the author block is filled in and in either language. Returns 1 on
    success, 0 if the heading was not found and the document is two columns throughout.
    """
    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf-8")

    paras = list(re.finditer(r"<w:p\b[^>]*>.*?</w:p>", doc, re.S))
    target = None
    for i, m in enumerate(paras):
        if _text(m.group(0)).strip().startswith(heading):
            target = paras[i - 1] if i else None
            break
    if target is None:
        return 0

    out = [doc[:target.start()], _with_section(target.group(0), ONE_COL),
           doc[target.end():]]

    tmp = docx.with_suffix(".tmp.docx")
    _rewrite_zip(docx, tmp, {"word/document.xml": "".join(out).encode("utf-8")})
    shutil.move(tmp, docx)
    return 1
