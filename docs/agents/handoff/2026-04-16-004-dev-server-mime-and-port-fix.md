# Handoff: Dev Server MIME Type and Port Conflict Fix

**Date:** 2026-04-16  
**Session:** 2026-04-16-004-dev-server-mime-and-port-fix

## What Was Accomplished

Fixed two dev server problems:

1. `.page` files (raw Markdown source) were being served as `application/octet-stream`, triggering a browser download instead of displaying as text.
2. `cabal run wiki -- serve` falsely claimed "Serving _site/ on http://127.0.0.1:8000" even when the port was already occupied, because Warp's default wildcard bind (`*:8000`) on IPv6 succeeds silently alongside an existing IPv4 `*:8000` listener — meaning two wiki processes could coexist undetected, with the browser always hitting the older IPv4 one.

### Files Changed

- **`Main.hs`** — MIME fix, port pre-check, better startup message, import additions
- **`wiki.cabal`** — added `network >= 3.1` to `build-depends`

### Key Changes in `Main.hs`

**MIME type for `.page` files:**
```haskell
ssGetMimeType = \file ->
    case takeExtension (T.unpack (fromPiece (fileName file))) of
        ""      -> return "text/html; charset=utf-8"
        ".page" -> return "text/plain; charset=utf-8"
        _       -> ssGetMimeType base file
```

**Port pre-check (probes both IPv4 and IPv6 before starting Warp):**
```haskell
checkPortFree :: Int -> IO ()
checkPortFree port = mapM_ check [(AF_INET,  SockAddrInet  (fromIntegral port) 0),
                                   (AF_INET6, SockAddrInet6 (fromIntegral port) 0 (0,0,0,0) 0)]
  where
    check (fam, addr) = do
        s <- socket fam Stream defaultProtocol
        result <- catch (bind s addr >> return True) (\(_ :: IOException) -> return False)
        close s
        if result then return ()
        else do
            putStrLn $ "Error: port " ++ show port ++ " already in use (" ++ show fam ++ ")."
            putStrLn $ "Kill existing servers:  kill $(lsof -n -P -i:" ++ show port ++ " | awk '/LISTEN/{print $2}')"
            exitFailure
```

**Startup message via `setBeforeMainLoop`** (only fires after socket is successfully bound — avoids the false "Serving..." print that appeared before Warp attempted the bind):
```haskell
warpSettings = setBeforeMainLoop
    (putStrLn $ "Serving _site/ on http://127.0.0.1:" ++ show port)
    $ setPort port defaultSettings
```

**New imports:**
```haskell
import Network.Wai.Handler.Warp (runSettings, defaultSettings, setPort, setHost, setBeforeMainLoop)
import Data.String (fromString)
import Network.Socket (socket, bind, close, Family(..), SocketType(..), defaultProtocol,
                       SockAddr(SockAddrInet, SockAddrInet6))
-- ScopedTypeVariables added to LANGUAGE pragma
```

## Key Decisions

### Why the MIME fix was hard to diagnose
`cabal build wiki` was persistently reporting "Up to date" even after source edits, which made it seem like changes weren't taking effect. Root cause: a stale wiki process was already running on port 8000 (IPv4), and every new `cabal run wiki -- serve` silently bound IPv6 instead. The browser always connected to the old IPv4 process, so the new binary's MIME fix was never exercised. The fix wasn't broken — it just wasn't being reached.

### macOS IPv4/IPv6 dual-listen quirk
On macOS, `AF_INET *:8000` and `AF_INET6 *:8000` are independent address families. The kernel does not consider them conflicting, so two processes can both `bind()` port 8000 without error. The browser connects via `127.0.0.1` (IPv4), so it always reaches the older process. The pre-check solution probes both families explicitly before starting Warp, so either conflict will be caught.

### Kill command targets LISTEN sockets only
`lsof -ti:8000` returns PIDs for *all* processes with any file descriptor touching port 8000 — including browser processes with open or recently-closed connections. The suggested kill command uses `awk '/LISTEN/'` to restrict to actual server processes.

### Cabal "Up to date" after edits
Cabal v2 uses content-addressed caching. `touch Main.hs` does not force a rebuild — only actual content changes do. `cabal clean && cabal build wiki` is the reliable way to force a full recompile when the cache is suspected to be stale.

## Important Context for Future Sessions

- **Workflow:** `cabal run wiki -- clean && cabal run wiki -- build` to regenerate `_site/`, then `cabal run wiki -- serve` to start the dev server. Do not use `rm -rf _site` — it removes output but leaves Hakyll's store cache, so the subsequent build skips regenerating pages it considers up-to-date.
- **Kill stale servers:** `kill $(lsof -n -P -i:8000 | awk '/LISTEN/{print $2}')` or `pkill -x wiki`.
- **Port default:** 8000. Override: `cabal run wiki -- serve 9000`.
- **`.page` files in `_site/`:** Hakyll's `"**.page"` static copy rule copies raw source files alongside the compiled HTML. This is pre-existing behavior (not introduced this session). The compiled output is extensionless (e.g. `_site/About`), the raw source copy is `_site/About.page`. Both exist simultaneously.
- **`ScopedTypeVariables` now enabled** in the `LANGUAGE` pragma — was needed for the `IOException` type annotation in the port pre-check catch handler.
- **`network` package** added to `wiki.cabal` `build-depends` (was a hidden transitive dep of warp before).
