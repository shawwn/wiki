#!/bin/sh
set -x
git checkout -- static/metadata/auto.hs
git checkout -- feed.rss
git pull
