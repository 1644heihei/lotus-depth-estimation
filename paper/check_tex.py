r"""Catch the LaTeX mistakes that only show up as a broken PDF: a \cite with no
\bibitem, a \ref with no \label, and the draft's own TODO markers."""
import io
import re
import sys

B = chr(92)
s = io.open(sys.argv[1], encoding="utf-8").read()

keys = set(re.findall(B + B + r"bibitem\{([^}]+)\}", s))
cited = set()
for m in re.findall(B + B + r"cite\{([^}]+)\}", s):
    cited |= {k.strip() for k in m.split(",")}
labels = set(re.findall(B + B + r"label\{([^}]+)\}", s))
used = set(re.findall(B + B + r"ref\{([^}]+)\}", s))

print(f"bibitems {len(keys)} | cited {len(cited)} | labels {len(labels)}")
print("cited but no bibitem :", sorted(cited - keys) or "none")
print("bibitem never cited  :", sorted(keys - cited) or "none")
print("ref with no label    :", sorted(used - labels) or "none")
print("TODO markers         :", s.count("TODO"))

body = s.split(B + r"begin{thebibliography}")[0]
words = re.sub(B + r"[a-zA-Z]+|[{}%&]", " ", body).split()
print(f"body words (rough)   : {len(words)}  "
      f"(~{len(words)/900:.1f} two-column pages of text)")
