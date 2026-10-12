r"""Check that main.tex and main_ja.tex carry the same numbers.

Two files holding the same results are two places for a number to be wrong in, and a
review of the English draft found several numbers that had drifted between sections of
one file alone. This compares the two languages: every decimal and every percentage in
one should appear in the other, except for the handful that differ only in how the
language writes them.

It compares how many times each number appears, not just whether it appears. A set
comparison missed a real error: the oracle's retention was corrected from 6.6% to 6.07%
in the English table and body but not in the English abstract, and not anywhere in the
Japanese file. Both numbers were then present in both files and the check passed while
three of the five sites were still wrong.
"""
import io
import re
import sys
from collections import Counter

# Figures that legitimately differ because the two languages spell them differently.
# The value is the count to forgive, or None to forgive any difference.
ALLOWED = {
    "868": None, "8.68": None,   # "868M" in English, "8.68 億" in Japanese
    "1.9": None,                 # "1.9M parameters" vs "193 万"
    "21.3": None,
}


def numbers(path):
    s = io.open(path, encoding="utf-8").read()
    s = s.split(r"\begin{thebibliography}")[0]          # references differ by design
    # The venue template forbids math in the abstract, so the English one spells
    # its numbers out ("the 6.07 percent", "one pixel") while the Japanese one
    # writes them. The two cannot match there, and comparing them only adds noise.
    if "\\end{abstract}" in s:
        s = s.split("\\end{abstract}", 1)[1]
    # A comment starts at an UNESCAPED %. Stripping at every % eats the rest of
    # any table row that contains \%, which silently hid half the numbers.
    s = re.sub(r"(?<!\\)%.*", "", s)
    out = Counter()
    # Decimals, grouped thousands, and whole-number percentages. The last of these was
    # missing: the share of the gain a coarse segmentation grid delivers was corrected
    # from 91% to 92% in the body and left at 91% in the abstract and the contribution
    # list, in both languages, and nothing here noticed because 91 has no decimal point.
    for m in re.findall(r"\d+(?:\{,\})?\d*\.\d+|\d+\{,\}\d{3}|\d+(?=\\%)", s):
        out[m.replace("{,}", ",")] += 1
    return out


def main():
    en, ja = numbers("main.tex"), numbers("main_ja.tex")
    missing, differing = [], []
    for k in sorted(set(en) | set(ja)):
        if k in ALLOWED:
            continue
        a, b = en[k], ja[k]
        if a == b:
            continue
        (missing if not a or not b else differing).append((k, a, b))
    print(f"main.tex {sum(en.values())} numbers | main_ja.tex {sum(ja.values())} numbers")
    for k, a, b in missing:
        print(f"  only in {'English' if b == 0 else 'Japanese'}: {k}")
    for k, a, b in differing:
        print(f"  count differs: {k} appears {a}x in English, {b}x in Japanese")
    if missing or differing:
        print("\nthe two versions disagree; fix before building either")
        return 1
    print("\nthe two versions agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
