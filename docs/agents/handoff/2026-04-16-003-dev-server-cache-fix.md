# Handoff: Dev Server Cache Fix

**Date:** 2026-04-16  
**Session:** 2026-04-16-003-dev-server-cache-fix

## What Was Accomplished

Fixed the dev server so that browsers never cache responses, preventing stale MIME-type downloads for extensionless pages like `/index`.

### Files Changed

- **`Main.hs`** — added `MaxAge(..)` to the `WaiAppStatic.Types` import; added `ssMaxAge = NoStore` to the `serveLocally` settings

### Change Summary

```haskell
-- Before
import WaiAppStatic.Types (StaticSettings(..), unsafeToPiece, fromPiece, fileName)
-- ...
, ssIndices = map unsafeToPiece ["index", "index.html", "index.htm"]
}

-- After
import WaiAppStatic.Types (StaticSettings(..), MaxAge(..), unsafeToPiece, fromPiece, fileName)
-- ...
, ssIndices = map unsafeToPiece ["index", "index.html", "index.htm"]
, ssMaxAge = NoStore
}
```

`NoStore` adds `Cache-Control: no-store` to every response from the dev server.

## Key Decisions

### Root cause: stale browser cache, not a code bug
Curl testing confirmed the server was already returning `Content-Type: text/html; charset=utf-8` for both `/` and `/index`. The browser was downloading `/index` because it had cached an earlier response (from before the MIME-type fix in session 002) with `Content-Type: application/octet-stream`. Without a `Cache-Control` directive, browsers are free to cache responses indefinitely based on `Last-Modified`.

### Why `NoStore` instead of `NoCache`
`NoStore` (`Cache-Control: no-store`) tells browsers to never write the response to any cache. `NoCache` (`Cache-Control: no-cache`) still caches but forces revalidation on every request. For a local dev server, full cache suppression (`NoStore`) is preferable — we always want fresh content, and the round-trip cost to localhost is negligible.

### `defaultFileServerSettings` default is `NoMaxAge`
The default (`ssMaxAge = NoMaxAge`) emits no `Cache-Control` header at all. Browsers interpret this as "cache at your discretion," which is why a stale response from a previous server run could persist across sessions.

## Important Context for Future Sessions

- The MIME-type fix (extensionless files served as `text/html`) was added in session 002 and is still in place; `ssMaxAge = NoStore` is additive.
- The dev server is started with `cabal run wiki -- serve [port]` (default port 8000). The `serve` subcommand is intercepted before Hakyll sees it; it does not go through any Hakyll code path.
- All other `StaticSettings` remain at `defaultFileServerSettings` defaults: `ssUseHash = False`, `ssAddTrailingSlash = False`, `ssRedirectToIndex = False`, `ss404Handler = Nothing`.
- `static/metadata/auto.hs` was also modified (new link metadata entries scraped during the session) but this is routine — it is not related to the server fix.
- If the browser still downloads after the fix is deployed, the immediate workaround is a hard refresh (Cmd+Shift+R in Chrome) or opening an incognito window to bypass the stale cache.
