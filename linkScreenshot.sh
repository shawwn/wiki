#!/bin/bash

# linkScreenshot.sh: screenshot URLs/PDFs
# Author: Gwern Branwen
# Date: 2019-08-25
# License: CC-0
#
# Shell script to screenshot URLs/PDFs for use with LinkMetadata.hs: we screenshot a path, optimize it, and save it to /static/previews/$SHA256($URL).png
#
# Example:
# $ linkScreenshot.sh 'http://forum.evageeks.org/post/464235/Evangelion-20-CRC-Anno-Interview/#464235' ~>
#
# Requires: wget, Google Chrome/Chromium, Ghostscript, ImageMagick, pngnq, and AdvPNG ('advancecomp')

set -e
set -x

t() { timeout --kill-after=60s 60s "$@"; }

pdf() {
    t gs -r300 -sDEVICE=pnggray -sOutputFile="$2" -dQUIET -dFirstPage=1 -dLastPage=1 -dNOPAUSE -dBATCH "$1"

    # crop down the PDF to eliminate the huge margins and focus on (probably) the title/abstract
    mogrify -gravity North -crop 85x80%+0+50 -gravity Center -scale 512x512 "$2"
    }

# we use SHA1 for turning a URL into an ID/hash because URL-encoding via `urlencode` causes Ghostscript to crash (and risks long-filename issues), MD5 isn't supported in the Subtle Crypto JS library most web browsers support, and SHA-256 & higher are just wastes of space in this context.
# WARNING: remember that 'echo' appends newlines by default!
HASH="$(echo -n "$@" | sha1sum - | cut -d ' ' -f 1).png"

# do we want to abort early if there is already a screenshot, or do we want to overwrite it anyway? (Perhaps a whole run went bad.)
INVALIDATE_CACHED_SCREENSHOT="false"
if [[ ! $CACHE_P == "true" && -s ~/wiki/static/previews/"$HASH" ]]; then
    exit 0
fi

# Local PDF:
if [[ "$@" =~ ^docs/.*\.pdf$ ]]; then
    pdf ~/wiki/"$@" /tmp/"$HASH"
else
    # Local HTML:
    if [[ "$@" =~ ^docs/.*\.html$ ]]; then
        t chromium-browser --disable-background-networking --disable-background-timer-throttling --disable-breakpad --disable-client-side-phishing-detection --disable-default-apps --disable-dev-shm-usage --disable-extensions --disable-features=site-per-process --disable-hang-monitor --disable-popup-blocking --disable-prompt-on-repost --disable-sync --disable-translate --metrics-recording-only --no-first-run --safebrowsing-disable-auto-update --enable-automation --password-store=basic --use-mock-keychain --hide-scrollbars --mute-audio --headless --disable-gpu --hide-scrollbars --screenshot=/tmp/"$HASH" \
          --window-size=512,512 "$@"
    else
       # Remote HTML, which might actually be a PDF:
       MIME=$(timeout 20s curl --write-out '%{content_type}' --silent --head -L -o /dev/null "$@")
       if [[ "$@" =~ .*\.pdf.* || "$MIME" == "application/pdf" || "$MIME" == "application/octet-stream" ]] ; then

           echo "Headless Chrome does not support PDF viewing (https://github.com/GoogleChrome/puppeteer/issues/299 \
                        https://github.com/GoogleChrome/puppeteer/issues/1872), so downloading instead..."

           t wget --quiet --output-file=/dev/null "$@" --output-document=/tmp/"$HASH".pdf
           pdf /tmp/"$HASH".pdf /tmp/"$HASH"
           rm /tmp/"$HASH".pdf
       else
           # Remote HTML, which is actually HTML:
           t chromium-browser --disable-background-networking --disable-background-timer-throttling --disable-breakpad --disable-client-side-phishing-detection --disable-default-apps --disable-dev-shm-usage --disable-extensions --disable-features=site-per-process --disable-hang-monitor --disable-popup-blocking --disable-prompt-on-repost --disable-sync --disable-translate --metrics-recording-only --no-first-run --safebrowsing-disable-auto-update --enable-automation --password-store=basic --use-mock-keychain --hide-scrollbars --mute-audio --headless --disable-gpu --hide-scrollbars --screenshot=/tmp/"$HASH" \
             --window-size=512,512 "$@"
       fi
    fi
fi

# Now that we have a PNG, somehow, optimize it so the bandwidth/storage isn't ruinous over thousands of hyperlinks:
pngnq -n 128 -s1 /tmp/"$HASH"
FILE="${HASH%.*}"
mv "/tmp/$FILE"-nq8.png /tmp/"$HASH"

advpng --iter 30 -z --shrink-insane /tmp/"$HASH"

mv /tmp/"$HASH" ~/wiki/static/previews/"$HASH"
