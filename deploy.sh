#!/bin/sh
git checkout -- static/metadata/auto.hs
git pull
exec ./sync.sh "$@"
