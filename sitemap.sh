#!/bin/sh
set -ex
echo "<?xml version=\"1.0\" encoding=\"UTF-8\"?> <urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"
find _site/ -not -name "*.page" -type f -size +2k -print0 | xargs -L 1 -0 ./urlencode.sh | sed -e 's/_site\/\(.*\)/\<url\>\<loc\>https:\/\/www\.shawwn\.com\1<\/loc><changefreq>monthly<\/changefreq><\/url>/'
echo "</urlset>"
