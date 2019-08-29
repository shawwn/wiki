#!/bin/sh
set -x
./predeploy.sh
exec ./sync.sh "$@"
