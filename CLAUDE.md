# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Haskell-based static site generator for a personal wiki (shawwn.net), forked from the Gwern.net engine. Content is written in Markdown (`.page` files), compiled by a Hakyll/Pandoc pipeline, and served from a Hetzner dedicated server via nginx.

## Toolchain Setup

GHC and cabal are managed via ghcup (installed at `~/.ghcup`). Source the env before using them:

```bash
source ~/.ghcup/env   # adds ghc, cabal to PATH
ghc --version         # 9.6.6
cabal --version       # 3.16.x
```

`cabal.project` sets `allow-newer: all` to resolve the old upper-bound constraints in transitive deps.

## Common Commands

```bash
# Compile the Haskell binary
cabal build wiki

# Full build (compile, clean, generate _site/, sitemap/RSS/MathJax)
./build.sh

# Generate site only (after binary is built)
cabal v2-run wiki -- build

# Clean generated site and cache
cabal v2-run wiki -- clean
```

## Architecture

### Build Pipeline (`build.sh`)
1. `./overlay.sh link` — symlinks private content from `../wiki-private` (no-ops if absent)
2. `cabal v2-build wiki` — compiles the Haskell generator
3. `cabal v2-run wiki -- clean` — wipes `_site/`
4. `cabal v2-run wiki -- build` — generates `_site/` via Hakyll
5. `./sitemap.sh`, `./feed.sh`, `./build-mathjax.sh` — post-processing
6. HTML validation via `tidy`

### Core Haskell Modules
- **`Main.hs`** — Hakyll site generator: compilation rules, templates, tags, RSS, redirects, Pandoc AST transformations
- **`LinkMetadata.hs`** — Link popup system: scrapes metadata (titles, abstracts, dates) from URLs via curl + tagsoup, caches results in `static/metadata/auto.hs` and `static/metadata/custom.hs`
- **`Inflation.hs`** — Rewrites dollar amounts in content with inflation-adjusted equivalents

### Content & Templates
- `*.page` — Markdown wiki articles (About.page, ML.page, Links.page, etc.)
- `static/templates/` — HTML templates (default.html, postitem.html, tags.html, analytics.html)
- `static/metadata/` — Cached link metadata (auto.hs = scraped, custom.hs = hand-curated)
- `static/css/`, `static/js/`, `static/img/`, `static/font/` — Static assets
- `docs/`, `images/` — Content assets referenced from `.page` files

### Deployment

The site runs on a Hetzner dedicated server (88.198.62.84, ssh alias `hetzner`).
Server setup details are in [sharc/docs/server-setup.md](https://github.com/shawwn/sharc/blob/main/docs/server-setup.md).

- **Serving**: nginx serves `_site/` directly with `default_type text/html` and `Cache-Control: no-cache`
- **Auto-rebuild**: a cron job (`/etc/cron.d/wiki-rebuild`) runs every 5 minutes as the `deploy` user, pulls both `wiki` and `wiki-private` repos, and rebuilds if either changed
- **Wiki location on server**: `/mnt/sda/private/wiki/`
- **Private content**: sibling repo `wiki-private` is overlayed via `overlay.sh link` at build time
- `env.sh` — Environment config (site URL, RSS metadata)

### Content Notes

- The `index` page is a hand-written HTML file (not a `.page`). It is copied as-is to `_site/`.
- The sidebar (logo + nav links) is hardcoded in two places: `static/templates/default.html` (for `.page` files) and `index` (for the home page). Keep them in sync.

### Key Dependencies
- Haskell: `hakyll 4.12`, `pandoc 2.7`, `tagsoup`, `aeson`, `arxiv`
- System tools: `imagemagick`, `tidy`, `mathjax-node-page`, GNU `parallel`, `ripgrep`
