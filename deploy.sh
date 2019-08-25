#!/bin/sh
git pull
exec ./sync.sh "$@"
