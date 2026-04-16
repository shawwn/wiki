# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Haskell-based static site generator for a personal wiki (shawwn.com), forked from the Gwern.net engine. Content is written in Markdown (`.page` files), compiled by a Hakyll/Pandoc pipeline, and deployed to S3.

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

# Full build (compile → clean → generate _site/ → sitemap/RSS/MathJax)
./build.sh

# Generate site only (after binary is built)
cabal v2-run wiki -- build

# Clean generated site and cache
cabal v2-run wiki -- clean

# Deploy to S3
./deploy.sh

# Start the webhook server (port 80, triggers deploys on GitHub push)
npm start
```

## Architecture

### Build Pipeline (`build.sh`)
1. `cabal v2-build wiki` — compiles the Haskell generator
2. `cabal v2-run wiki -- clean` — wipes `_site/`
3. `cabal v2-run wiki -- build` — generates `_site/` via Hakyll
4. `./sitemap.sh`, `./feed.sh`, `./build-mathjax.sh` — post-processing
5. HTML validation via `tidy`, then `s3cmd sync` to S3

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
- `index.js` — Express.js webhook server on port 80; GitHub push events trigger `./deploy.sh`
- `env.sh` — Environment config (S3 bucket name, site URL, RSS metadata)

### Key Dependencies
- Haskell: `hakyll 4.12`, `pandoc 2.7`, `tagsoup`, `aeson`, `arxiv`
- Node.js: `express` (webhook server only)
- System tools: `imagemagick`, `s3cmd`, `tidy`, `mathjax-node-page`, GNU `parallel`, `ripgrep`
