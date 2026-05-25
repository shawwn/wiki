# Handoff: Private Content Overlay via wiki-private Symlinks

**Date:** 2026-05-25
**Session:** 2026-05-25-002-private-content-overlay

## What Was Accomplished

Added support for overlaying private content from a sibling `wiki-private` repo onto the public wiki at build time. Private `.page` files, images, docs, and other assets appear in the built site alongside public content without leaking into the public repo.

### Files Changed

- **`overlay.sh`** (new) - Shell script with `link`, `unlink`, `status` subcommands. Creates symlinks from `../wiki-private/` (or `$WIKI_PRIVATE_DIR`) into the public repo, tracks them in a manifest, and adds entries to `.git/info/exclude`.
- **`LinkMetadata.hs`** - `readLinkMetadata` now optionally loads `static/metadata/custom-private.hs` if it exists (via `doesFileExist`), merging private link metadata with public.
- **`Main.hs`** - Added `System.Directory (doesFileExist)` import, `readRedirectsIfExists` helper, and optional loading of `static/redirects/Redirects-private.hs`.
- **`build.sh`** - Calls `./overlay.sh link` at the start (no-ops if no private repo found).
- **`predeploy.sh`** - Calls `./overlay.sh unlink` at the top to ensure clean public deploys.
- **`.gitignore`** - Added `.private-overlay-manifest`, `static/metadata/custom-private.hs`, `static/redirects/Redirects-private.hs`.

### How It Works

1. `overlay.sh link` walks `../wiki-private/`, creates symlinks for each file at the corresponding path in the public repo
2. Hakyll's glob patterns (`"**.page"`, `"images/**"`, `"docs/**"`, etc.) pick up symlinked files transparently
3. Private metadata and redirects use separate filenames (`custom-private.hs`, `Redirects-private.hs`) loaded optionally by the Haskell code
4. `overlay.sh unlink` removes all symlinks, cleans up empty directories, and removes `.git/info/exclude` entries

## Key Decisions

### Symlinks over copying
Symlinks are cleaner than copying because there is no risk of accidentally committing copies, and changes in the private repo are reflected immediately (important for watch mode). Hakyll follows symlinks via standard Haskell filesystem functions.

### Persistent overlay (no cleanup trap in build.sh)
The overlay stays active between builds rather than being set up and torn down each time. This supports iterative development and watch mode. `predeploy.sh` handles cleanup for deploys.

### Separate files for private metadata/redirects
Private metadata goes in `custom-private.hs` (not `custom.hs`) and private redirects in `Redirects-private.hs` (not `Redirects.hs`). This avoids file conflicts and lets the Haskell code merge them at load time.

### Conflict detection
If a private file has the same name as an existing public file, `overlay.sh` warns and skips rather than overwriting.

## Important Context for Future Sessions

- **Private repo location**: defaults to `../wiki-private`, configurable via `$WIKI_PRIVATE_DIR` env var.
- **Private repo structure**: must mirror the public repo's directory layout. Pages at root, assets in `images/`, `docs/`, `static/`, etc.
- **Private `.page` files need full frontmatter**: `title`, `description`, `tags`, `created`, `status`, `confidence`, `importance` (Hakyll's template fails without these).
- **A test private repo exists** at `../wiki-private/` with `test-private.page` and `docs/private/test.txt`.
- **`auto.hs` mutation**: during builds, Hakyll writes discovered link metadata to `static/metadata/auto.hs`. This will include entries from private content. `predeploy.sh` resets it via `git checkout` before deploys.
- **All guard paths tested**: building without `../wiki-private` works normally; every integration point checks existence first and no-ops gracefully.
