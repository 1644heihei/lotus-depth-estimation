#!/bin/sh
# main_ja.tex needs LuaLaTeX: pdflatex cannot set Japanese. Same two passes, same
# three numbers that decide whether a layout is sound.
set -e
export PATH="$PATH:/c/Users/nihei/AppData/Local/Programs/MiKTeX/miktex/bin/x64"
cd "$(dirname "$0")"
for pass in 1 2; do
  lualatex -interaction=nonstopmode main_ja.tex > _ja.log 2>&1 || {
    echo "pass $pass FAILED"; grep -A4 "^!" _ja.log | head -20; exit 1; }
done
echo "pages: $(grep -o 'Output written on main_ja.pdf ([0-9]* page' _ja.log | grep -o '[0-9]*')"
echo "undefined references: $(grep -c undefined _ja.log || true)"
echo "overfull boxes: $(grep -c 'Overfull' _ja.log || true)"
rm -f _ja.log
