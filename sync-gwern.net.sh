#!/bin/bash

# key dependencies: GHC, Hakyll, s3cmd, emacs, curl, tidy (HTML5 version), urlencode ('gridsite-clients' package)

cd ~/wiki/

if [[ -n $(which ghc) && -n $(which s3cmd) && -n $(which git) && -n $(which emacs) && -n $(which curl) && -n $(which ping) && -n $(which tidy) && -n $(which linkchecker) && -n $(which du) && -n $(which rm) && -n $(which find) && -n $(which fdupes) && -n $(which urlencode) && -n $(which xargs) && -n $(which sed) ]]
then
    set -ex
    # Cleanup pre:
    rm --recursive --force -- ~/wiki/_cache/ ~/wiki/_site/ ./hakyll ./*.o ./*.hi || true

    # Build:
    ## gwern.net is big and Hakyll+Pandoc is slow, so it's worth the hassle of compiling an optimized version to build
    ghc -Wall -optl-fuse-ld=gold -rtsopts -threaded -O2 --make hakyll.hs
    ./hakyll build +RTS -N8 -RTS
    ## generate a sitemap file for search engines:
    ## possible alternative implementation in hakyll: https://www.rohanjain.in/hakyll-sitemap/
    (echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?> <urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"
     find _site/ -not -name "*.page" -type f -size +2k -print0 | xargs -L 1 -0 urlencode -m |
         sed -e 's/_site\/\(.*\)/\<url\>\<loc\>https:\/\/www\.gwern\.net\/\1<\/loc><changefreq>monthly<\/changefreq><\/url>/'

     echo "</urlset>") >> ./_site/sitemap.xml

    ## use https://github.com/pkra/mathjax-node-page/ to statically compile the MathJax rendering of the MathML to display math instantly on page load
    ## background: https://joashc.github.io/posts/2015-09-14-prerender-mathjax.html ; installation: `npm install --prefix ~/src/ mathjax-node-page`
    staticCompileMathJax () {
        fgrep --quiet "<math " "$@" && \
            TARGET=$(mktemp /tmp/XXXXXXX.html) && \
            cat "$@" | ~/src/node_modules/mathjax-node-page/bin/mjpage --output CommonHTML  >> "$TARGET" && \
            mv "$TARGET" "$@" && echo "$@ succeeded" # || echo $@ 'failed MathJax compilation!';
        }
    export -f staticCompileMathJax
    find ./ -path ./_site -prune -type f -o -name "*.page" | sort | sed -e 's/\.page//' -e 's/\.\/\(.*\)/_site\/\1/' | parallel --progress staticCompileMathJax || true

    rm -- ./hakyll ./*.o ./*.hi || true

    # Testing compilation results:
    ## is there a sane number & size of compiled files?
    [ "$(find ./_site/ -type f | wc --lines)" -ge 10500 ]
    [ "$(du --summarize --total --bytes ./_site/ | tail -1 | cut --field=1)" -ge 15000000000 ]

    ## Any stray interwiki links?
    find ./_site/ -type f -not -name "*.page" -exec grep --quiet -I . {} \; -print0 | xargs -0 -n 5000 fgrep -- '(!Wikipedia)' | sort | fgrep '</' | grep --color=always Wikipedia || true
    ## do various files validate as HTML5?
    # tidy -quiet -errors --doctype html5 ./_site/index ./_site/About ./_site/Links ./_site/Coin-flip &>/dev/null # disabled due to slash problem in collapsing code - htmltidy disagrees with the W3C validator. TODO: file a bug report on tidy?
    set +x; IFS=$(echo -en "\n\b");
    PAGES="$(find . -type f -name "*.page" | grep -v "_site/" | sort -u)"
    for PAGE in $PAGES; do
        HTML="${PAGE%.page}"
        echo -n "$HTML: "
        (tidy -quiet -errors --doctype html5 ./_site/"$HTML" 2>&1 >/dev/null | fgrep -v -e '<link> proprietary attribute ' -e 'Warning: trimming empty <span>') || true
    done; set -x

    ## Is the Internet up?
    ping -q -c 5 google.com  &> /dev/null

    # Sync:
    ## use the cheapest S3 storage & heavy caching to save money & bandwidth:
    s3() { s3cmd -v -v --human-readable-sizes --reduced-redundancy --add-header="Cache-Control: max-age=7776000, public" "$@"; }
    TARGET="s3://www.gwern.net"
    s3 --guess-mime-type --default-mime-type="text/html" --delete-removed sync ./_site/ "$TARGET"/
    ## force CSS to be synced with CSS mimetype and not as "text/html"; TODO: does the s3cmd CSS bug still exist as of 2019?
    s3 --mime-type="text/css" put ./_site/static/css/*.css "$TARGET"/static/css/
    s3 --mime-type="text/css" put ./_site/docs/statistics/order/beanmachine-multistage/*.css "$TARGET"/docs/statistics/order/beanmachine-multistage/
    s3 --mime-type="text/css" put ./_site/docs/gwern.net-gitstats/gitstats.css "$TARGET"/docs/gwern.net-gitstats/

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
else
    echo "binaries missing"
fi
