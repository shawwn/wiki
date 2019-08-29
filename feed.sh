#!/bin/bash
set -e
source env.sh
if command -v html-beautify >/dev/null
then
  python3 gen_rss.py | html-beautify
else
  python3 gen_rss.py
fi
