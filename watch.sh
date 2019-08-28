#!/bin/bash
set -ex

source env.sh

./build.sh

cabal v2-run wiki -- watch --no-server &

sleep 10

while sleep 1 ; do find _site -name '*.page' -type f | entr -d ./build-mathjax.sh ; done

