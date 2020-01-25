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

url="$1"
shift 1

export PATH="/usr/local/bin:$PATH"

t() { timeout --kill-after=60s 60s "$@"; }

pdf() {
    t gs -r300 -sDEVICE=pnggray -sOutputFile="$2" -dQUIET -dFirstPage=1 -dLastPage=1 -dNOPAUSE -dBATCH "$1"

    # crop down the PDF to eliminate the huge margins and focus on (probably) the title/abstract
    mogrify -gravity Center -crop 90x85%+0+50 -gravity Center -scale 1582x2048 "$2"
    }

imgur() {
    img="$(./uploadScreenshot.sh "$@" | jq -r .data.link)"
    echo "$img" > "${@}.txt"
    echo "<img src=\"${img}\">" > "${@}.html"
    }

# we use SHA1 for turning a URL into an ID/hash because URL-encoding via `urlencode` causes Ghostscript to crash (and risks long-filename issues), MD5 isn't supported in the Subtle Crypto JS library most web browsers support, and SHA-256 & higher are just wastes of space in this context.
# WARNING: remember that 'echo' appends newlines by default!
HASH="$(echo -n "$url" | sha1sum - | cut -d ' ' -f 1).png"
dst="${1:-static/previews/${HASH}}"

# do we want to abort early if there is already a screenshot, or do we want to overwrite it anyway? (Perhaps a whole run went bad.)
INVALIDATE_CACHED_SCREENSHOT="false"
if [[ ! $CACHE_P == "true" && -s "${dst}" ]]; then
    if [[ ! -s "${dst}.html" ]]; then
      imgur "${dst}"
    fi
    exit 0
fi

# Local PDF:
if [[ "$url" =~ ^docs/.*\.pdf$ ]]; then
    pdf "$url" /tmp/"$HASH"
else
    # Local HTML:
    if [[ "$url" =~ ^docs/.*\.html$ ]]; then
        t chromium-browser --disable-background-networking --disable-background-timer-throttling --disable-breakpad --disable-client-side-phishing-detection --disable-default-apps --disable-dev-shm-usage --disable-extensions --disable-features=site-per-process --disable-hang-monitor --disable-popup-blocking --disable-prompt-on-repost --disable-sync --disable-translate --metrics-recording-only --no-first-run --safebrowsing-disable-auto-update --enable-automation --password-store=basic --use-mock-keychain --hide-scrollbars --mute-audio --headless --disable-gpu --hide-scrollbars --screenshot=/tmp/"$HASH" \
          --window-size=791,1024 "$url"
    else
       # Remote HTML, which might actually be a PDF:
       MIME=$(timeout 20s curl --write-out '%{content_type}' --silent --head -L -o /dev/null "$url")
       if [[ "$url" =~ .*\.pdf.* || "$MIME" == "application/pdf" || "$MIME" == "application/octet-stream" ]] ; then

           echo "Headless Chrome does not support PDF viewing (https://github.com/GoogleChrome/puppeteer/issues/299 \
                        https://github.com/GoogleChrome/puppeteer/issues/1872), so downloading instead..."

           t wget -U "Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1)" --quiet --output-file=/dev/null "$url" --output-document=/tmp/"$HASH".pdf
           pdf /tmp/"$HASH".pdf /tmp/"$HASH"
           rm /tmp/"$HASH".pdf
       else
           # Remote HTML, which is actually HTML:
           t chromium-browser --disable-background-networking --disable-background-timer-throttling --disable-breakpad --disable-client-side-phishing-detection --disable-default-apps --disable-dev-shm-usage --disable-extensions --disable-features=site-per-process --disable-hang-monitor --disable-popup-blocking --disable-prompt-on-repost --disable-sync --disable-translate --metrics-recording-only --no-first-run --safebrowsing-disable-auto-update --enable-automation --password-store=basic --use-mock-keychain --hide-scrollbars --mute-audio --headless --disable-gpu --hide-scrollbars --screenshot=/tmp/"$HASH" \
             --window-size=791,1024 "$url"
       fi
    fi
fi

# Now that we have a PNG, somehow, optimize it so the bandwidth/storage isn't ruinous over thousands of hyperlinks:
#pngnq -n 128 -s1 /tmp/"$HASH"
FILE="${HASH%.*}"
#mv "/tmp/$FILE"-nq8.png /tmp/"$HASH"

#advpng --iter 30 -z --shrink-insane /tmp/"$HASH"

mv /tmp/"$HASH" "$dst"

imgur "${dst}"
