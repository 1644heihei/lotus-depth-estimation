r"""Check that main.tex and main_ja.tex carry the same numbers.

Two files holding the same results are two places for a number to be wrong in, and a
review of the English draft found several numbers that had drifted between sections of
one file alone. This compares the two languages: every decimal and every percentage in
one should appear in the other, except for the handful that differ only in how the
language writes them.
"""
import io
import re
import sys

# Figures that legitimately differ because the two languages spell them differently.
ALLOWED = {
    "868", "8.68",   # "868M" in English, "8.68 億" in Japanese
    "1.9",           # "1.9M parameters" vs "193 万"
    "21.3",
}


def numbers(path):
    s = io.open(path, encoding="utf-8").read()
    s = s.split(r"\begin{thebibliography}")[0]          # references differ by design
    # A comment starts at an UNESCAPED %. Stripping at every % eats the rest of
    # any table row that contains \%, which silently hid half the numbers.
    s = re.sub(r"(?<!\\)%.*", "", s)
    out = set()
    for m in re.findall(r"\d+(?:\{,\})?\d*\.\d+|\d+\{,\}\d{3}", s):
        out.add(m.replace("{,}", ","))
    return out


def main():
    en, ja = numbers("main.tex"), numbers("main_ja.tex")
    only_en = sorted(x for x in en - ja if x not in ALLOWED)
    only_ja = sorted(x for x in ja - en if x not in ALLOWED)
    print(f"main.tex {len(en)} numbers | main_ja.tex {len(ja)} numbers")
    print(f"only in English : {only_en or 'none'}")
    print(f"only in Japanese: {only_ja or 'none'}")
    if only_en or only_ja:
        print("\nthe two versions disagree; fix before building either")
        return 1
    print("\nthe two versions agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
