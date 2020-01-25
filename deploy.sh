#!/bin/sh
set -x
./predeploy.sh
./build.sh
./sync.sh
