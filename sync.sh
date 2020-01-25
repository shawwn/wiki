#!/bin/bash
set -ex

source env.sh

## Is the Internet up?
ping -q -c 1 google.com  &> /dev/null

# Sync:
## use the cheapest S3 storage & heavy caching to save money & bandwidth:
s3() { s3cmd -v -v --human-readable-sizes --reduced-redundancy --add-header="Cache-Control: max-age=$CACHE_MAX_AGE, public" "$@"; }
s3 --guess-mime-type --default-mime-type="text/html" --delete-removed sync ./_site/ "$BUCKET"/
## force CSS to be synced with CSS mimetype and not as "text/html"; TODO: does the s3cmd CSS bug still exist as of 2019?
s3 --mime-type="text/css" put ./_site/static/css/*.css "$BUCKET"/static/css/
#s3 --mime-type="application/javascript" put ./_site/static/js/*.js "$BUCKET"/static/js/
#s3 --mime-type="text/css" put ./_site/docs/statistics/order/beanmachine-multistage/*.css "$BUCKET"/docs/statistics/order/beanmachine-multistage/
#s3 --mime-type="text/css" put ./_site/docs/gwern.net-gitstats/gitstats.css "$BUCKET"/docs/gwern.net-gitstats/

exit 0

set +x
# Testing post-sync:
## Is gwern.net and serving gzipped HTTP successfully?
c() { curl --compressed --silent --output /dev/null --head "$@"; }
[ "$(c --write-out '%{http_code}' 'https://www.gwern.net/index')" -eq 200 ]
## did any of the key pages mysteriously vanish from the live version?
linkchecker --check-extern -r1 'https://www.gwern.net/index'
## are some of the live MIME types correct?
[ "$(c --write-out '%{content_type}' 'https://www.gwern.net/index')"                  == "text/html" ]
[ "$(c --write-out '%{content_type}' 'https://www.gwern.net/static/css/default.css')" == "text/css" ]
[ "$(c --write-out '%{content_type}' 'https://www.gwern.net/images/logo.png')"        == "image/png" ]
[ "$(c --write-out '%{content_type}' 'https://www.gwern.net/docs/history/1694-gregory.pdf')"  == "application/pdf" ]
## known-content check:
curl --silent 'https://www.gwern.net/index' | fgrep --quiet 'This is the website of <strong>Gwern Branwen</strong>'
curl --silent 'https://www.gwern.net/Zeo'   | fgrep --quiet 'lithium orotate'
## - traffic checks/alerts are done in Google Analytics: alerts on <900 pageviews/daily, <40s average session length/daily.
## - latency/downtime checks are done in Pingdom [TODO: replacement now that Pingdom is killing free plans]

# Cleanup post:
rm --recursive --force -- ~/wiki/_cache/ ~/wiki/_site/ || true

# Testing files, post-sync
## check for duplicate files:
fdupes --quiet --recurse --sameline --size --nohidden ~/wiki/ || true
## check for read-only outside ./.git/ (weird but happened):
find . -perm u=r -path '.git' -prune
## check for corrupted documents:
find ./docs/ -type f -name "*.pdf"  -exec file {} \; | fgrep -v 'PDF document'
find ./ -type f -name "*.jpg" -exec file {} \; | fgrep -v 'JPEG image data'
find ./ -type f -name "*.png" -exec file {} \; | fgrep -v 'PNG image data'
## check for 'encrypted' PDFs:
checkEncryption () { ENCRYPTION=$(exiftool -q -q -Encryption "$@"); if [ "$ENCRYPTION" != "" ]; then echo "Encrypted: $@"; fi; }
export -f checkEncryption
find ./docs/ -type f -name "*.pdf" | parallel checkEncryption
## DjVu is deprecated (due to SEO - no search engines will crawl DjVu, turns out!):
find ./ -type f -name "*.djvu"
## GIF is deprecated; check for GIFs which should be converted to WebMs/MP4:
find ./ -type f -name "*.gif" | fgrep -v -e 'static/img' -e 'docs/gwern.net-gitstats/'
## Find JPGs which haven't been compressed to <=60% quality:
find ./ -type f -name "*.jpg" | xargs -n 5000 identify -format '%Q %F\n'| sort --numeric-sort | egrep -v -e '^[0-6][0-9] '
## Find JPGS which are too wide (1800px is almost an entire screen width, which is too large):
for IMAGE in `find /home/gwern/wiki/images/ -type f -name "*.jpg" -or -name "*.png" | sort`; do
SIZEW=$(identify -format "%w" "$IMAGE")
if (( $SIZEW > 1800  )); then echo "$IMAGE" $SIZEW; fi;
done

# if the first of the month, download all pages and check that they have the right MIME type and are not suspiciously small or redirects.
if [ $(date +"%d") == "1" ]; then

PAGES=$(cd ~/wiki/ && find . -type f -name "*.page" | sed -e 's/\.\///' -e 's/\.page$//' | sort)
c() { curl --compressed --silent --output /dev/null --head "$@"; }
for PAGE in $PAGES; do
    MIME=$(c --max-redirs 0 --write-out '%{content_type}' "https://www.gwern.net/$PAGE")
    if [ "$MIME" != "text/html" ]; then echo "$PAGE: $MIME"; exit 2; fi

    SIZE=$(curl --max-redirs 0 --compressed --silent "https://www.gwern.net/$PAGE" | wc --bytes)
    if [ "$SIZE" -lt 7500 ]; then echo "$PAGE : $SIZE : $MIME" && exit 2; fi
done
fi

echo "Sync successful"
