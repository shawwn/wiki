#!/bin/sh
set -x
exec ./predeploy.sh
exec ./sync.sh "$@"
