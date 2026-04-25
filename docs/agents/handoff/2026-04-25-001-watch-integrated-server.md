# Handoff: Watch Command Integrated Server

**Date:** 2026-04-25  
**Session:** 2026-04-25-001-watch-integrated-server

## What Was Accomplished

Merged the dev server into `cabal run wiki -- watch` so only one command is needed for local development. Previously the workflow required two terminals: `watch --no-server` (file watcher) and `serve` (custom Warp server). Now `cabal run wiki -- watch` does both.

### Files Changed

- **`wiki.cabal`** — added `async >= 2.2` to `build-depends`
- **`Main.hs`** — new imports, refactored server into `serveWithoutCheck`, extracted `siteRules`, added `watch` case in `main`, `SO_REUSEADDR` fix in `checkPortFree`
- **`README.md`** — "Serve Locally" section simplified to single command

### Key Changes in `Main.hs`

**Server logic split into two functions** (so the watch path can fork the server without re-running the port check):
```haskell
serveWithoutCheck :: Int -> IO ()   -- just runs Warp
serveWithoutCheck port = ...

serveLocally :: Int -> IO ()        -- port check + server
serveLocally port = checkPortFree port >> serveWithoutCheck port
```

**`siteRules :: Rules ()` extracted as top-level** to avoid duplicating the large `hakyll $ do ...` block across the `watch` and `_` branches.

**`watch` case in `main`:**
```haskell
("watch":rest) -> do
    let port = case filter (/= "--no-server") rest of { [p] -> read p; _ -> 8000 }
    checkPortFree port
    withAsync (serveWithoutCheck port) $ \_ ->
        withArgs ["watch", "--no-server"] $ hakyll siteRules
```

`withAsync` (from the `async` package) starts the Warp server in a background thread. When Hakyll exits — whether normally or via Ctrl+C — `withAsync`'s `bracket` cancels the server thread, which sends `AsyncCancelled` into Warp, causing it to close the listening socket cleanly.

**`SO_REUSEADDR` added to `checkPortFree`:**
```haskell
check (fam, addr) = do
    s <- socket fam Stream defaultProtocol
    setSocketOption s ReuseAddr 1     -- added
    result <- catch (bind s addr >> return True) (\(_ :: IOException) -> return False)
    ...
```

Without this, `checkPortFree` would false-alarm on TIME_WAIT connections left over from the previous run's accepted HTTP connections. Warp itself sets `SO_REUSEADDR` on its listening socket, so the check now matches Warp's actual behavior: fails only on an active LISTEN conflict, not on TIME_WAIT residue.

**New imports:**
```haskell
import Control.Concurrent.Async (withAsync)
import System.Environment (lookupEnv, getArgs, withArgs)  -- withArgs added
import Network.Socket (..., setSocketOption, SocketOption(ReuseAddr))  -- added
```

## Key Decisions

### `withAsync` over `forkIO`
`forkIO` was the first instinct but the user specifically asked for proper cleanup on parent exit. `withAsync` wraps the server in `bracket (async ...) cancel`, so any exit path — normal, exception, or Ctrl+C — cancels the server thread. Warp handles `AsyncCancelled` cleanly via its own internal `bracket` for the server socket.

### `withArgs` to override Hakyll's arg parsing
Hakyll reads `getArgs` internally; there is no `hakyllWithArgs` API in the version used. `System.Environment.withArgs` temporarily replaces the process args for the duration of the `hakyll` call, so Hakyll sees `["watch", "--no-server"]` and runs its file watcher without starting its own inferior built-in server.

### `SO_REUSEADDR` scope
Added only to `checkPortFree`, not to `serveWithoutCheck` (Warp already sets it). The fix is targeted: `checkPortFree` was the only thing behaving inconsistently relative to Warp.

### TIME_WAIT vs CLOSE_WAIT
The intermittent "port in use" error after Ctrl+C was caused by TIME_WAIT connections on port 8000 (server side, local port 8000 → Chrome's ephemeral port). These are invisible in `lsof` after the process exits (kernel-only state) but still block `bind()` without `SO_REUSEADDR`. The CLOSE_WAIT connections shown by `lsof` (Chrome's ephemeral ports → 8000) were a red herring — those are on Chrome's ports and cannot block binding to 8000.

## Important Context for Future Sessions

- **New dev workflow:** `cabal run wiki -- watch` — starts both the Hakyll file watcher and the custom Warp server on port 8000. Single terminal, single command.
- **`serve` still works:** `cabal run wiki -- serve` is unchanged and can still be used standalone (e.g. to serve a pre-built `_site/` without watching).
- **Port override:** `cabal run wiki -- watch 9000` (the port arg is stripped of `--no-server` then parsed as the port).
- **`async` package** added to `wiki.cabal` `build-depends` — was not previously a listed dependency.
- **`siteRules`** is now a top-level `Rules ()` function (lines ~92–140 in `Main.hs`); `main` is much shorter. Both `watch` and the catch-all branch call `hakyll siteRules`.
- **Ctrl+C behavior:** The Warp server is properly cancelled via `withAsync`. Subsequent `cabal run wiki -- watch` should succeed immediately without "port in use" errors, thanks to the `SO_REUSEADDR` fix.
