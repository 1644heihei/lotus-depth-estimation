r"""Fit the converted paper into the venue's own Word template.

The template (conference-template-a4.docx) carries the layout and a named style for
every part of an IEEE paper. pandoc matches a reference document's styles by name, so
passing the template as --reference-doc already lands the headings, the abstract and the
author block on the right ones. What it cannot do is the rest:

  * pandoc's own style names for the parts the template calls something else - a title is
    "Title" to pandoc and "paper title" here - so those are remapped afterwards.
  * columns, because pandoc copies only the reference document's LAST section, which in
    this template is the one-column tail rather than the two-column body.
  * the references heading, which LaTeX prints from \begin{thebibliography} and pandoc
    does not emit at all.

Geometry is the template's own body section, converted from its points to the twips the
schema wants: A4 at 595.3 by 841.9 pt, 54 pt top, 45.35 pt sides, 72 pt bottom, two
columns 18 pt apart.
"""
from __future__ import annotations

import re
import shutil
import zipfile
from pathlib import Path

from docx_layout import CONTINUOUS, _rewrite_zip, _text, _with_section

A4_PAGE = ('<w:pgSz w:w="11906" w:h="16838" w:code="9"/>'
           '<w:pgMar w:top="1080" w:right="907" w:bottom="1440" w:left="907"'
           ' w:header="720" w:footer="720" w:gutter="0"/>')
A4_TWO_COL = '<w:cols w:num="2" w:space="360" w:equalWidth="1"/>'
A4_ONE_COL = '<w:cols w:num="1" w:space="360"/>'

# pandoc's style ids on the left, the template's on the right. Anything not listed keeps
# the id pandoc gave it, which for headings and the abstract is already the template's.
STYLE_MAP = {
    "Title": "papertitle",
    "Subtitle": "papersubtitle",
    "AbstractTitle": "Abstract",
    "ImageCaption": "figurecaption",
    # "table head", not the figure style: both auto-number, and sending a table caption
    # to the figure style numbers it as a figure. A table captioned "Fig. 8" is worse
    # than one captioned nothing.
    "TableCaption": "tablehead",
    "FirstParagraph": "a3",          # Body Text
    "Compact": "a3",
    "BodyText": "a3",
    "SourceCode": "a3",
    "BlockText": "a3",
}


def normalise(template: Path, out: Path) -> tuple[Path, int]:
    """Rewrite the template's measurements as the integers the schema requires.

    The file as supplied expresses every length in points - w:pgSz w="595.30pt",
    w:spacing w:after="6pt", hundreds of them across document, styles and numbering -
    where OOXML wants twentieths of a point as an integer. Word repairs this when it
    opens the template itself, but pandoc copies the values through verbatim and the
    result is a file Word refuses, even for a two-line document. Converting them first
    is what makes the template usable as a reference document at all.
    """
    members = ("word/document.xml", "word/styles.xml", "word/numbering.xml",
               "word/settings.xml", "word/footer1.xml", "word/footnotes.xml",
               "word/endnotes.xml")
    fixed, changes = 0, {}
    with zipfile.ZipFile(template) as z:
        names = set(z.namelist())
        for name in members:
            if name not in names:
                continue
            text = z.read(name).decode("utf-8")

            def to_twips(m: re.Match) -> str:
                nonlocal fixed
                fixed += 1
                return f'{m.group(1)}="{round(float(m.group(2)) * 20)}"'

            new = re.sub(r'(\w+)="(-?\d+(?:\.\d+)?)pt"', to_twips, text)
            if name == "word/document.xml":
                # pandoc copies the trailing section verbatim but brings none of the
                # parts it points at, so the footerReference in the template's last
                # section arrives with no footer1.xml and no relationship behind it.
                # Word calls the result corrupt, for a two-line document as readily as
                # for this one. Dropping the references is what makes it open.
                new = re.sub(r"<w:(?:header|footer)Reference\b[^>]*/>", "", new)
            if new != text:
                changes[name] = new.encode("utf-8")
    _rewrite_zip(template, out, changes)
    return out, fixed


def localise_numbering(docx: Path, labels: dict[str, tuple[str, str]]) -> int:
    """Rewrite the caption auto-numbering the template defines.

    The template is an English one: its figure captions number themselves "Fig. %1." and
    its tables "TABLE %1. " in Roman. A Japanese paper whose body says 図 1 and 表 6
    should not caption them Fig. 1 and TABLE VI, and the Roman numerals disagree with
    the cross-references pandoc writes besides. `labels` maps a numId to the lvlText and
    numFmt its first level should use.
    """
    with zipfile.ZipFile(docx) as z:
        if "word/numbering.xml" not in z.namelist():
            return 0
        nb = z.read("word/numbering.xml").decode("utf-8")

    num_to_abs = {m.group(1): m.group(2) for m in re.finditer(
        r'<w:num w:numId="(\d+)"[^>]*>\s*<w:abstractNumId w:val="(\d+)"', nb)}
    done = 0
    for num_id, (text, fmt) in labels.items():
        abs_id = num_to_abs.get(num_id)
        if abs_id is None:
            continue
        m = re.search(rf'<w:abstractNum w:abstractNumId="{abs_id}"[^>]*>.*?</w:abstractNum>',
                      nb, re.S)
        if not m:
            continue
        block = m.group(0)
        new = re.sub(r'<w:lvlText w:val="[^"]*"\s*/>',
                     f'<w:lvlText w:val="{text}"/>', block, count=1)
        new = re.sub(r'<w:numFmt w:val="[^"]*"\s*/>',
                     f'<w:numFmt w:val="{fmt}"/>', new, count=1)
        if new != block:
            nb = nb.replace(block, new, 1)
            done += 1

    tmp = docx.with_suffix(".tmp.docx")
    _rewrite_zip(docx, tmp, {"word/numbering.xml": nb.encode("utf-8")})
    shutil.move(tmp, docx)
    return done


def drop_paragraph(docx: Path, text: str) -> int:
    """Delete paragraphs whose entire text is `text`.

    \\begin{thebibliography}{00} takes a width argument, and pandoc renders it as a
    paragraph reading "00" just above the references.
    """
    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf-8")
    out, pos, n = [], 0, 0
    for m in re.finditer(r"<w:p\b[^>]*>.*?</w:p>", doc, re.S):
        if _text(m.group(0)).strip() != text:
            continue
        out.append(doc[pos:m.start()])
        pos = m.end()
        n += 1
    if not n:
        return 0
    out.append(doc[pos:])
    _replace_document(docx, "".join(out))
    return n


def remap_styles(docx: Path, mapping: dict[str, str] = None) -> int:
    """Point every paragraph at the template's style for its role."""
    mapping = STYLE_MAP if mapping is None else mapping
    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf-8")

    hits = 0

    def one(m: re.Match) -> str:
        nonlocal hits
        want = mapping.get(m.group(1))
        if not want:
            return m.group(0)
        hits += 1
        return f'<w:pStyle w:val="{want}"/>'

    doc = re.sub(r'<w:pStyle w:val="([^"]+)"\s*/>', one, doc)
    _replace_document(docx, doc)
    return hits


def style_bibliography(docx: Path, n_items: int, heading: str,
                       heading_style: str = "1", item_style: str = "references") -> int:
    """Put the last `n_items` paragraphs on the references style, under a heading.

    Counted from the end rather than found by text: the bibliography is the last thing in
    the document and every entry is one paragraph, so the count from \\bibitem is exact,
    while matching on an author's name would break the moment a reference changed.
    """
    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf-8")

    paras = [m for m in re.finditer(r"<w:p\b[^>]*>.*?</w:p>", doc, re.S)
             if _text(m.group(0)).strip()]
    if len(paras) < n_items:
        return 0
    items = paras[-n_items:]

    out, pos = [], 0
    for i, m in enumerate(items):
        out.append(doc[pos:m.start()])
        if i == 0:
            out.append(f'<w:p><w:pPr><w:pStyle w:val="{heading_style}"/></w:pPr>'
                       f'<w:r><w:t xml:space="preserve">{heading}</w:t></w:r></w:p>')
        out.append(_restyle(m.group(0), item_style))
        pos = m.end()
    out.append(doc[pos:])
    _replace_document(docx, "".join(out))
    return n_items


def _restyle(para: str, style: str) -> str:
    if "<w:pStyle" in para:
        return re.sub(r'<w:pStyle w:val="[^"]+"\s*/>',
                      f'<w:pStyle w:val="{style}"/>', para, count=1)
    if "<w:pPr>" in para:
        return para.replace("<w:pPr>", f'<w:pPr><w:pStyle w:val="{style}"/>', 1)
    return re.sub(r"(<w:p\b[^>]*>)",
                  r"\1" + f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>', para, count=1)


def _replace_document(docx: Path, doc: str) -> None:
    tmp = docx.with_suffix(".tmp.docx")
    _rewrite_zip(docx, tmp, {"word/document.xml": doc.encode("utf-8")})
    shutil.move(tmp, docx)


def apply_a4_sections(docx: Path, wide_figures: list[int],
                      heading: str = "Abstract") -> tuple[int, int]:
    """The template's column layout: title block full width, body in two, wide figures
    full width again. Returns (title spanned, figures spanned)."""
    with zipfile.ZipFile(docx) as z:
        doc = z.read("word/document.xml").decode("utf-8")

    # the trailing section is the body: two columns on A4
    doc = _set_geometry(doc, A4_TWO_COL)

    paras = list(re.finditer(r"<w:p\b[^>]*>.*?</w:p>", doc, re.S))
    edits: dict[int, str] = {}

    for i, m in enumerate(paras):
        if _text(m.group(0)).strip().startswith(heading) and i:
            edits[i - 1] = A4_ONE_COL
            break

    images = [i for i, m in enumerate(paras) if "<a:blip" in m.group(0)]
    for n in wide_figures:
        if n < len(images) and images[n]:
            edits.setdefault(images[n] - 1, A4_TWO_COL)
            edits[images[n]] = A4_ONE_COL

    out, pos = [], 0
    for i in sorted(edits):
        m = paras[i]
        out.append(doc[pos:m.start()])
        out.append(_with_section(m.group(0), edits[i], page=A4_PAGE))
        pos = m.end()
    out.append(doc[pos:])
    _replace_document(docx, "".join(out))
    spanned_figs = sum(1 for v in edits.values() if v is A4_ONE_COL) - 1
    return (1 if edits else 0), max(spanned_figs, 0)


def _set_geometry(doc: str, cols: str) -> str:
    """Set the page and columns of the document's trailing section, in place.

    The obvious move - prepending the geometry right after <w:sectPr> - produces a file
    Word calls corrupt. CT_SectPr is a sequence, and the template's trailing section
    opens with a footerReference, so inserting pgSz ahead of it puts the reference out
    of order. Replacing the elements the section already has keeps the order it had.
    """
    new = {"pgSz": re.search(r"<w:pgSz\b[^>]*/>", A4_PAGE).group(0),
           "pgMar": re.search(r"<w:pgMar\b[^>]*/>", A4_PAGE).group(0),
           "cols": cols}

    def one(m: re.Match) -> str:
        s = m.group(0)
        for tag, repl in new.items():
            s, n = re.subn(rf"<w:{tag}\b[^>]*/>", repl.replace("\\", "\\\\"), s, count=1)
            if not n:                      # absent: put it where the schema wants it
                anchor = {"pgSz": r"(<w:type\b[^>]*/>)", "pgMar": r"(<w:pgSz\b[^>]*/>)",
                          "cols": r"(<w:pgMar\b[^>]*/>)"}[tag]
                s, n = re.subn(anchor, r"\1" + repl.replace("\\", "\\\\"), s, count=1)
                if not n:
                    s = s.replace(">", ">" + repl, 1)
        return s

    sections = list(re.finditer(r"<w:sectPr\b.*?</w:sectPr>", doc, re.S))
    if not sections:
        return doc
    last = sections[-1]
    return doc[:last.start()] + one(last) + doc[last.end():]
