#!/bin/sh
set -x
./overlay.sh unlink
git checkout -- static/metadata/auto.hs
git checkout -- feed.rss
git pull
