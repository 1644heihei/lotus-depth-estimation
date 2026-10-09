#!/bin/sh
# HTML -> PDF without a LaTeX or pandoc install: headless Edge prints the page
# with the @media print rules in main_ja.html (A4, 20mm margins).
# Chrome works identically if Edge is not present.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
EDGE="/c/Program Files (x86)/Microsoft/Edge/Application/msedge.exe"
[ -f "$EDGE" ] || EDGE="/c/Program Files/Google/Chrome/Application/chrome.exe"
"$EDGE" --headless --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="$(cygpath -w "$HERE/main_ja.pdf")" \
  "file:///$(cygpath -m "$HERE/main_ja.html")"
echo "-> $HERE/main_ja.pdf"
