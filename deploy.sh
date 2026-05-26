#!/bin/bash
set -ex

PRIVATE_DIR="${WIKI_PRIVATE_DIR:-../wiki-private}"
REMOTE_WIKI=/mnt/sda/private/wiki
REMOTE_PRIVATE=/mnt/sda/private/wiki-private
SERVER=hetzner

# Clean up local overlay and build artifacts before pushing
./predeploy.sh

# Push both repos
git push
if [ -d "$PRIVATE_DIR/.git" ]; then
  git -C "$PRIVATE_DIR" push
fi

# Pull, re-overlay, and rebuild on server
ssh $SERVER "set -ex
  cd $REMOTE_WIKI && git pull origin main
  cd $REMOTE_PRIVATE && git pull origin main
  cd $REMOTE_WIKI
  source /mnt/sda/private/.ghcup/env
  export TMPDIR=$REMOTE_WIKI/tmp
  ./overlay.sh link
  ./build.sh
"

# Old S3 deployment (Gwern's original flow):
#   ./predeploy.sh
#   exec ./sync.sh "$@"
