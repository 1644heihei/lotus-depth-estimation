#!/bin/sh
# Compile main.tex with MiKTeX. Two passes, because the cross-references need them.
# Reports the page count and anything that would show as a layout fault in the PDF.
set -e
export PATH="$PATH:/c/Users/nihei/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
cd "$(dirname "$0")"
for pass in 1 2; do
  pdflatex -interaction=nonstopmode main.tex > _build.log 2>&1 || {
    echo "pass $pass FAILED"; grep -A4 "^!" _build.log | head -20; exit 1; }
done
echo "pages: $(grep -o 'Output written on main.pdf ([0-9]* pages' _build.log | grep -o '[0-9]*')  (limit 6)"
echo "undefined references: $(grep -c undefined _build.log || true)"
echo "overfull boxes: $(grep -c 'Overfull' _build.log || true)"
rm -f _build.log
