# Handoff: Haskell Build Modernization

**Date:** 2026-04-16  
**Session:** 2026-04-16-001-haskell-build-modernization

## What Was Accomplished

Got the wiki's Haskell static site generator building on Apple Silicon macOS (arm64, macOS 14.4.1) for the first time since ~2020. The codebase was pinned to GHC 8.6.x which never supported arm64.

### Files Changed

- **`wiki.cabal`** — rewrote all version bounds for GHC 9.6; removed `HTTP` and `arxiv` packages
- **`cabal.project`** — new file; sets `allow-newer: all` to bypass stale transitive upper bounds
- **`Main.hs`** — ported from pandoc 2.7 API to pandoc 3.9 API
- **`LinkMetadata.hs`** — same pandoc port; inlined `arxiv` package helpers
- **`Inflation.hs`** — same pandoc port
- **`CLAUDE.md`** — created; documents toolchain setup, commands, and architecture

### Toolchain Installed

```bash
# GHC 9.6.6 + cabal 3.16.1.0 via ghcup
~/.ghcup/bin/ghc      # 9.6.6
~/.ghcup/bin/cabal    # 3.16.1.0

# Must source before use:
source ~/.ghcup/env
```

Packages resolved to: **pandoc 3.9.0.2**, **hakyll 4.16.8.0**, and their full dependency trees (~100 packages compiled from source into `~/.cabal/store/`).

### Build Verified

```
cabal build wiki      # succeeds
cabal run wiki -- --help   # shows hakyll CLI
```

## Key Decisions

### GHC Version: 9.6.6
Chose 9.6.6 (not 9.8 or 9.10) as it's a stable LTS-era release with confirmed hakyll/pandoc support. The original `base <4.13` constraint would have required GHC 8.6.x, which never had arm64 binaries.

### `allow-newer: all` in cabal.project
Many transitive deps still carry stale upper bounds (e.g. `MissingH`, `filestore`). Rather than patching each one, `allow-newer: all` lets cabal's solver ignore upper bounds globally. The build succeeds — these bounds were conservative, not load-bearing.

### Removed `arxiv` package (0.0.1, 2015)
The package was last updated in 2015 and would not compile with modern GHC. Only 5 functions were used (`getTitle`, `getSummary`, `getUpdated`, `getDoi`, `getAuthorNames`), all simple TagSoup XML helpers. These were inlined directly into `LinkMetadata.hs`.

### Removed `HTTP` package
Used only for `Network.HTTP.urlEncode` in `convertInterwikiLinks`. Replaced with the already-present `escape` function defined in `Main.hs`.

### pandoc 2.7 → 3.9 API Changes Fixed

| Old API | New API |
|---|---|
| `Str String` | `Str Text` — use `T.unpack`/`T.pack` at boundaries |
| `Code _ String` | `Code _ Text` — same |
| `Link _ _ (String, String)` | `Link _ _ (Text, Text)` — use `T.isInfixOf`, `(<>)`, `T.pack` |
| `Attr = (String, [String], [(String, String)])` | `Attr = (Text, [Text], [(Text, Text)])` |
| `bottomUp f` | `walk f` from `Text.Pandoc.Walk` |
| `writerTemplate :: Maybe String` | `Maybe (Template Text)` — compile with `compileTemplate` |
| `writerHtmlQTags` field | Removed in pandoc 3; dropped |

### Template Compilation
`writerTemplate` now requires a compiled `Template Text` (from `doctemplates`). Compiling happens inside the hakyll `compile` block via `unsafeCompiler`:

```haskell
templ <- unsafeCompiler $ do
    result <- compileTemplate "" ("<div id=\"TOC\">$toc$</div>\n<div id=\"markdownBody\">$body$</div>" :: T.Text)
    case result of
        Right t -> return t
        Left e  -> error e
```

`compileTemplate` is polymorphic over `Monad m` and works directly in `IO` — do **not** wrap with `runIO` (that double-wraps the `Either` and breaks type inference).

## Context for Future Sessions

### The Binary
```bash
source ~/.ghcup/env
cabal run wiki -- build    # generates _site/
cabal run wiki -- clean    # wipes _site/ and cache
```

### What Has NOT Been Tested Yet
- Actually running `cabal run wiki -- build` against the real content (`.page` files). The binary compiles and shows its help, but the full site generation pipeline (Pandoc transforms, link metadata scraping, image dimension detection via ImageMagick) has not been exercised.
- `build.sh` / `deploy.sh` — these assume `s3cmd`, `imagemagick`, `tidy`, `mathjax-node-page` are installed.
- The `static/metadata/custom.hs` and `static/metadata/auto.hs` files are required at runtime by `readLinkMetadata`.

### Known Warnings (non-fatal)
- `LICENSE` file missing (cabal warns but builds fine)
- Several pattern matches in `LinkMetadata.hs` are non-exhaustive (original code; harmless in practice as the HTML structure is predictable)

### String/Text Boundary Strategy
Internal logic in all three modules stays as `String`. Conversion happens only at the pandoc `Inline`/`Block` boundary: `T.unpack` going in, `T.pack` going out. `MetadataItem` type remains `(String, String, String, String, String)` — values are `T.pack`'d when inserted into pandoc `Attr`.
