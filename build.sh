#!/bin/bash
set -ex

source env.sh

# Cleanup pre:
#rm --recursive --force -- ~/wiki/_cache/ ~/wiki/_site/ ./hakyll ./*.o ./*.hi || true

# Build:
## gwern.net is big and Hakyll+Pandoc is slow, so it's worth the hassle of compiling an optimized version to build
#ghc -Wall -optl-fuse-ld=gold -rtsopts -threaded -O2 --make hakyll.hs
#./hakyll build +RTS -N8 -RTS

cabal v2-build wiki
cabal v2-run wiki -- clean
cabal v2-run wiki -- build

#rm -f ./hakyll
#ghc -Wall -optl-fuse-ld=gold -rtsopts -threaded -O2 --make hakyll.hs
#./hakyll build +RTS -N8 -RTS --verbose

## generate a sitemap file for search engines:
## possible alternative implementation in hakyll: https://www.rohanjain.in/hakyll-sitemap/
./sitemap.sh > ./_site/sitemap.xml
cat ./_site/sitemap.xml

## generate an rss feed
./feed.sh > ./_site/feed.rss
cat ./_site/feed.rss

## convert mathml
./build-mathjax.sh

#rm -- ./hakyll ./*.o ./*.hi || true

# Testing compilation results:
## is there a sane number & size of compiled files?
#[ "$(find ./_site/ -type f | wc --lines)" -ge 10500 ]
#[ "$(du --summarize --total --bytes ./_site/ | tail -1 | cut --field=1)" -ge 15000000000 ]

## Any stray interwiki links?
#find ./_site/ -type f -not -name "*.page" -exec grep --quiet -I . {} \; -print0 | xargs -0 -n 5000 fgrep -- '(!Wikipedia)' | sort | fgrep '</' | grep --color=always Wikipedia || true
## do various files validate as HTML5?
# tidy -quiet -errors --doctype html5 ./_site/index ./_site/About ./_site/Links ./_site/Coin-flip &>/dev/null # disabled due to slash problem in collapsing code - htmltidy disagrees with the W3C validator. TODO: file a bug report on tidy?
set +x;
IFS=$(echo -en "\n\b");
PAGES="$(find . -type f -name "*.page" | grep -v "_site/" | sort -u)"
for PAGE in $PAGES; do
HTML="${PAGE%.page}"
[ -f "_site/${HTML}.html" ] && HTML="${HTML}.html"
echo -n "$HTML: "
(tidy -quiet -errors --doctype html5 ./_site/"$HTML" 2>&1 >/dev/null | fgrep -v -e '<link> proprietary attribute ' -e 'Warning: trimming empty <span>') || true
done; set -x

