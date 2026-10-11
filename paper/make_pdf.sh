#!/bin/sh
# HTML -> PDF without a LaTeX or pandoc install: headless Edge prints the page with the
# @media print rules in the file (A4, 20mm margins). Chrome works identically.
#
#   ./make_pdf.sh            both
#   ./make_pdf.sh main_en    just the English one
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
[ -f "$EDGE" ] || EDGE="/c/Program Files/Google/Chrome/Application/chrome.exe"
for name in ${*:-main_ja main_en}; do
  "$EDGE" --headless --disable-gpu --no-pdf-header-footer \
    --print-to-pdf="$(cygpath -w "$HERE/$name.pdf")" \
    "file:///$(cygpath -m "$HERE/$name.html")" 2>/dev/null
  echo "-> $HERE/$name.pdf"
done
