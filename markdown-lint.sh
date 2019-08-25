#!/bin/bash
# see https://www.gwern.net/About#markdown-checker

set +x

fgp () { grep -F --context=1 --line-number --color=always "$@"; }
egp () { grep -E --context=1 --line-number --color=always "$@"; }

for PAGE in "$@"
do
    if [[ $PAGE == *.page ]]; then

        # warn if not text, perhaps due to bad copy-paste
        file "$PAGE" | fgp --invert-match 'text';

        # find bad URLS, unacceptable/unreliable/risky domains, malformed syntax, unmatched apostrophes
        fgp -e 'http://dl.dropbox' -e 'http://news.ycombinator.com' -e 'http://github.com' \
            -e 'http://www.coursera.org' -e '.wiley.com/' -e 'http://www.ncbi.nlm.nih.gov/pubmed/' \
            -e 'www.tandfonline.com/doi/abs/' -e 'jstor.org' -e 'springer.com' -e 'springerlink.com' \
            -e 'www.mendeley.com' -e 'academia.edu' -e 'researchgate.net' -e 'pdf.yt' \
            -e 'photobucket' -e 'imgur.com' -e 'hathitrust.org' -e 'emilkirkegaard.dk' -e 'arthurjensen.net' \
            -e 'humanvarieties.org' -e 'libgen.io/' -e 'gen.lib.rus.ec/' -e 'sci-hub.bz/' -e '](http://www.scilogs.com/' \
            -e 'sci-hub.cc/' -e "papers.nber.org/" -e '](!wikipedia' -e 'https://wwww.' -e 'http://wwww.' \
            -e 'arxiv.org/pdf/' -e 'http://33bits.org' -e 'https://gwern.net' -e 'https://gwern.net' -e 'web.archive.org/web/2' \
            -e 'webarchive.org.uk/wayback/' -e 'webcitation.org' -e 'plus.google.com' -e 'www.deepdotweb.com' -e 'raikoth.net' \
            -e 'drive.google.com/file' -e 'ssrn.com' -e 'ardenm.us' -e 'gnxp.nofe.me' -e 'psycnet.apa.org' \
            -e 'wellcomelibrary.org/item/' -e 'dlcs.io/pdf/' -e 'secure.wikimedia.org' -e 'http://en.wikipedia.org' \
            -e 'http://biorxiv.org' -e 'https://biorxiv.org' -e 'http://www.biorxiv.org' -e 'http://arxiv.org' -- "$PAGE";
        # check for aggregator-hosted PDFs and host them on gwern.net to make them visible to Google Scholar/provide backups:
        link-extractor.hs "$PAGE" | egp --only-matching -e '^http://.*archive\.org/.*\.pdf$';
        egp -e 'http://www.pnas.org/content/.*/.*/.*.abstract' -e '[^\.]t\.test\(' -e '^\~\~\{\.' -- "$PAGE";
        # look for broken syntax:
        fgp -e '(www' -e ')www' -e '![](' -e ']()' -e '](/wiki/' -e '](wiki/' -e '——–' -e '——' -e '↔' \
            -e ' percent ' -e "    Pearson'" -e '~~~{.sh}' -e 'library("' -e ' +-' -e ' -+' -e '"collapse Summary"'-- "$PAGE";

        # look for personal uses of illegitimate statistics & weasel words, but filter out blockquotes
        fgp -e ' significant ' -e ' significantly ' -e ' obvious' -e 'basically' -e ' the the ' -- "$PAGE" | egrep --invert-match '[[:space:]]*>';

        # look for English/imperial units as a reminder to switch to metric as much as possible:
        fgp -e ' feet' -e ' foot ' -e ' pound ' -e ' mile ' -e ' miles ' -e ' inch' -- "$PAGE"

        # check for duplicate footnote IDs (force no highlighting, because the terminal escape codes trigger bracket-matching)
        grep -E --only-matching '^\[\^.*\]: ' -- "$PAGE" | sort | uniq --count | \
            grep -F --invert-match '      1 [^';

        # image hotlinking deprecated; impolite, and slows page loads & site compiles
        egp --only-matching '\!\[.*\]\(http://.*\)' -- "$PAGE";
        # indicates broken copy-paste of image location
        egp --only-matching '\!\[.*\]\(wiki/.*\)' -- "$PAGE";

        # look for images used without newline in between them; in some situations, this leads to odd distortions
        # of aspect ratio/zooming or something (first discovered in Correlation.page in blockquotes)
        # egrep --perl-regexp --null --only-matching -e '\!\[.*\]\(.*\)\n\!\[.*\]\(.*\)' -- "$PAGE";
        grep --perl-regexp --null --only-matching -e '\!\[.*\]\(.*\)\n\!\[.*\]\(.*\)' -- "$PAGE";

        # look for unescaped single dollar-signs (risk of future breakage)
        egp '^[^$]* [^\"]\$[^$]*$' -- "$PAGE";

        # instead of writing 'x = ~y', unicode as '≈'
        fgp -e '= ~' -- "$PAGE" | fgp --invert-match ' mods'
        fgp -e '?!' -e '!?' -e '<->' -- "$PAGE"

        [ "$(grep -E '^description: ' "$PAGE" | wc --char)" -ge 90 ] || echo 'Description metadata too short.'

        markdown-length-checker.hs "$PAGE";
        markdown-footnote-length.hs "$PAGE";
        proselint "$PAGE";

        # look for syntax errors making it to the final HTML output:
        HTML=$(mktemp  --suffix=".html")
        cat "$PAGE" | pandoc --metadata lang=en --metadata title="$1" --mathml --to=html5 --standalone --number-sections --toc --reference-links --css=https://www.gwern.net/static/css/default.css -f markdown+smart --template=/home/gwern/bin/bin/pandoc-template-html5-articleedit.html5  - --output="$HTML"
        tidy -quiet -errors --doctype html5 "$HTML";
        fgp -e "<""del"">" "$HTML";
        elinks -dump --force-html "$HTML" \
                     | fgp -e '\frac' -e '\times' -e '(http' -e ')http' -e '[http' -e ']http'  \
                           -e ' _ ' -e '[^' -e '^]' -e '/* ' -e ' */' -e '<!--' -e '-->' -e '<-- ' -e '<—' -e '—>' \
                           -e '$title$' -e '<del>' -e '.smallcaps' -e '</q<' -e '<q>' -e '</q>' \
                           -e '$description$' -e '$author$' -e '$tags$' -e '$category$' \
                           -e '(!Wikipedia' -e '(!Hoogle' -e 'http://www.gwern.net' -e 'http://gwern.net' ; # ))

        link-extractor.hs "$PAGE" | egrep -v -e "^http" -e '^!Wikipedia' -e '^#' -e '^/'
        link-extractor.hs "$PAGE" | egrep -v -e '^!Wikipedia$' | sort | uniq --count | sort --numeric-sort | egrep -v -e '.* 1 '

        # we use link annotations on URLs to warn readers about PDFs; if a URL ends in 'pdf', it gets a PDF icon. What about URLs which redirect to or serve PDF?
        # we must manually annotate them with a '#pdf'. Check URLs in a page for URLs which serve a PDF MIME type but do not mention PDF in their URL:
        checkPDF() {
            MIME=$(timeout 30s curl --silent -I -l "$@" | fgrep -i -e "Content-Type: application/pdf" -e "Content-Type: application/octet-stream")
            if [ ${#MIME} -gt 5 ]; then
                if [[ ! $@ =~ .*pdf.* ]] && [[ ! $@ =~ .*PDF.* ]]; then
                 echo "UNWARNED PDF: " "$@" "$MIME"
                fi;
           fi; }
        export -f checkPDF
        # examples: no/no/yes
        ## checkPDF 'http://www.nytimes.com/2009/11/15/business/economy/15view.html ' # no
        ## checkPDF 'http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.208.2314&rep=rep1&type=pdf' # yes
        ## checkPDF 'https://files.osf.io/v1/resources/np2jd/providers/osfstorage/59614dec594d9002288271b6?action=download&version=1&direct' # yes
        link-extractor.hs "$PAGE" | egrep "^http" | sort -u | shuf | parallel -n 1 checkPDF

        # Check linked PDFs for missing OCR and missing metadata fields:
        for PDF in $(link-extractor.hs "$PAGE" | egrep -e '^/docs/' -e 'https:\/\/www\.gwern\.net\/' | \
                                                 egrep '\.pdf$' | sed -e 's/\/docs/docs/' -e 's/https:\/\/www\.gwern\.net\///' ); do

          TITLE=$(exiftool -printFormat '$Title' -Title "$PDF")
          AUTHOR=$(exiftool -printFormat '$Author' -Author "$PDF")
          DATE=$(exiftool -printFormat '$Date' -Date "$PDF")
          DOI=$(exiftool -printFormat '$DOI' -DOI "$PDF")
          TEXT_LENGTH=$(pdftotext $PDF - | wc --chars)

          if (( $TEXT_LENGTH < 1024 )); then
              echo "$PDF OCR text length: $TEXT_LENGTH"
          fi
          if [[ -z $TITLE || -z $AUTHOR || -z $DATE || -z $DOI ]]; then
             exiftool -Title -Author -Date -DOI "$PDF"
          fi
        done

        # Finally, check for broken external links; ignore local URLs which are usually either relative links to
        # other pages in the repo or may be interwiki links like '!Wikipedia'.
        linkchecker --no-status --check-extern --threads=1 --timeout=20 -r1 --ignore='file://.*' "$HTML"
    fi
    if [[ $PAGE == *.sh ]]; then
        shellcheck "$PAGE"
    fi
    if [[ $PAGE == *.hs ]]; then
        hlint "$PAGE"
    fi
    if [[ $PAGE == *.html ]]; then
        tidy -quiet -errors --doctype html5 "$PAGE"
    fi
done
