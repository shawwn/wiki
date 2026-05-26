#!/bin/sh
set -ex
./overlay.sh unlink
git checkout -- static/metadata/auto.hs 2>/dev/null || true
git checkout -- feed.rss 2>/dev/null || true
git pull
