# Handoff: Rename Domain shawwn.com to shawwn.net

**Date:** 2026-05-25  
**Session:** 2026-05-25-001-rename-domain-shawwn-net

## What Was Accomplished

Renamed all references to `shawwn.com` to `shawwn.net` across the entire repo after the original domain was lost to a squatter.

### Files Changed (text replacements)

- **`env.sh`** - S3 bucket, website hostname, RSS URL, RSS title
- **`robots.txt`** - sitemap URL
- **`About.page`** - description and body text (3 occurrences)
- **`Changelog.page`** - description
- **`Links.page`** - body text
- **`LinkMetadata.hs`** - URL normalization in `annotateLink`, link dispatcher prefix match, and curl target in `gwern` scraper (3 occurrences)
- **`README.md`** - grep example command
- **`CLAUDE.md`** - project overview
- **`static/templates/default.html`** - creator meta, contact URL, canonical URL, apple-mobile-web-app-title, application-name
- **`static/templates/tags.html`** - description meta, canonical URL, apple-mobile-web-app-title, application-name
- **`static/404.html`** - apple-mobile-web-app-title, application-name, Google fixurl site
- **`static/css/default.css`** - CSS selectors for internal link styling (2 occurrences, both in commented and active blocks)
- **`static/css/www.shawwn.com/static/css/default.css`** - same CSS selectors (2 occurrences)
- **`static/metadata/auto.hs`** - About page description string
- **`static/metadata/custom.hs`** - newsletter description and changelog link URL
- **`docs/agents/handoff/2026-04-25-004-handoff-page-and-changelog.md`** - handoff doc text

### Directories Renamed

- `static/css/www.shawwn.com/` -> `static/css/www.shawwn.net/`
- `static/css/inspector%3a/www.shawwn.com/` -> `static/css/inspector%3a/www.shawwn.net/`

## Key Decisions

### Global replace with no exceptions
Every occurrence of `shawwn.com` was replaced, including in handoff docs and metadata cache files. The old domain is gone to a squatter, so there is no reason to preserve any reference to it.

### Skipped `_site/` directory
The `_site/` directory is generated output and will be rebuilt on the next `cabal v2-run wiki -- build`. Only source files were modified.

### LinkMetadata.hs string length constant
The `linkDispatcher` function in `LinkMetadata.hs:80` uses `drop 22 l` to strip the domain prefix. Both `https://www.shawwn.com/` and `https://www.shawwn.net/` are exactly 22 characters, so the constant did not need updating.

## Important Context for Future Sessions

- **DNS/hosting not updated here.** This session only changed source code references. The S3 bucket, CloudFlare config, and any DNS records still need to be updated separately to actually serve from `shawwn.net`.
- **The `static/css/www.shawwn.net/` directory** is a browser-inspector-captured copy of the site's CSS. It may have been created by Safari's Web Inspector and is checked into the repo. Its purpose is unclear but it was preserved and renamed.
- **No build was run.** The Haskell binary and `_site/` were not rebuilt. A full `./build.sh` should be run before deploying to verify everything compiles and generates correctly with the new domain.
