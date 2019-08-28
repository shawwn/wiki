#!/bin/bash
set -ex

source env.sh

## use https://github.com/pkra/mathjax-node-page/ to statically compile the MathJax rendering of the MathML to display math instantly on page load
## background: https://joashc.github.io/posts/2015-09-14-prerender-mathjax.html ; installation: `npm install --prefix ~/src/ mathjax-node-page`
staticCompileMathJax () {
[ -f "$@" ] && fgrep --quiet "<math " "$@" && \
    TARGET="$@.tmp" && \
    cat "$@" | mjpage --output CommonHTML  >> "$TARGET" && \
    mv "$TARGET" "$@" && echo "$@ succeeded" # || echo $@ 'failed MathJax compilation!';
}
export -f staticCompileMathJax
find ./ -path ./_site -prune -type f -o -name "*.page" | sort | sed -e 's/\.page//' -e 's/\.\/\(.*\)/_site\/\1/' | xargs -n 1 -I {} bash -c 'staticCompileMathJax "$@"; staticCompileMathJax "$@.html"' _ {} || true


