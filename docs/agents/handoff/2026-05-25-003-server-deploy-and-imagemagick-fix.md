# Handoff: ImageMagick remote URL fix (2026-05-25)

**Date:** 2026-05-25
**Session:** 2026-05-25-003-server-deploy-and-imagemagick-fix

## What Was Accomplished

### ImageMagick remote URL fix

**Commit:** `a8a5556` - Skip remote URLs in staticImg

`staticImg` in Main.hs runs ImageMagick's `identify` to get image
dimensions for `<img>` tags that lack height/width attributes. It
already skipped SVGs and data URIs via the `vector` guard, but not
remote URLs (http/https). When a page referenced an external image
(e.g. `https://i.imgur.com/M7DJPXW.png` in memorybox.page),
`identify` failed, `error` was thrown, and Hakyll caught the exception
and wrote a 0-byte output file.

Fix: added http/https prefix checks to the `vector` guard in
`staticImg` (line 350 of Main.hs), so remote URLs are returned
unchanged.

### Files Changed

- **`Main.hs`** (line 350) - Added `|| ("http://" \`isPrefixOf\` ...) || ("https://" \`isPrefixOf\` ...)` to the `vector` guard in `staticImg`

### Branch rename

Default branch renamed from `master` to `main` via GitHub API.

## Key Decisions

Skipping remote images entirely (rather than downloading them and
running `identify`) matches the existing pattern for SVGs/data URIs
and avoids network dependencies during builds. The images still render
in the browser; they just won't have explicit height/width attributes
in the HTML.

## Important Context for Future Sessions

- Pages that had 0-byte outputs (memorybox, swarm) should now build
  correctly. If other pages produce 0-byte outputs, check for external
  image references without height/width attributes.
- The wiki is now deployed on a Hetzner server with auto-rebuild.
  Server setup details are tracked in the sharc repo, not here.
