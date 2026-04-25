# Handoff: Live Reload via SSE

**Date:** 2026-04-25  
**Session:** 2026-04-25-002-live-reload-sse

## What Was Accomplished

Added integrated live reload to `cabal run wiki -- watch`. When a `.page` file is edited and Hakyll rebuilds `_site/`, all connected browsers automatically reload and restore their scroll position. No extra processes or ports needed.

### Files Changed

- **`wiki.cabal`** — added `wai >=3.2`, `wai-extra >=3.1`, `fsnotify >=0.4`, `stm >=2.5`, `http-types >=0.12` to `build-depends` (all were already transitively installed)
- **`Main.hs`** — new imports, `startSiteWatcher`, `liveReloadMiddleware`, `injectReloadScript`, `injectIntoHtml`, `reloadScriptTag` helpers; updated `serveWithoutCheck` signature; updated `serveLocally`; updated `watch` branch in `main`

### Key Changes in `Main.hs`

**`startSiteWatcher`** — watches `_site/` with `fsnotify`, debounces events via a `TVar Int` generation counter, broadcasts a `ServerEvent` reload signal after 300ms of quiet. Sends SSE heartbeat comments every 25s to keep browser connections alive:

```haskell
startSiteWatcher :: Chan ServerEvent -> IO ()
startSiteWatcher chan = withManager $ \mgr -> do
    lastRef <- newTVarIO (0 :: Int)
    void $ watchTree mgr "_site" (const True) $ \ev ->
        case ev of
            Modified {} -> fire lastRef
            Added    {} -> fire lastRef
            _           -> return ()
    forever $ threadDelay 25000000 >> writeChan chan (CommentEvent "keepalive")
  where
    fire lastRef = void $ forkIO $ do
        gen <- atomically $ do
            n <- readTVar lastRef
            writeTVar lastRef (n + 1)
            return (n + 1)
        threadDelay 300000
        current <- atomically (readTVar lastRef)
        when (current == gen) $
            writeChan chan (ServerEvent (Just "reload") Nothing ["reload"])
```

**`liveReloadMiddleware`** — WAI middleware that routes `GET /_reload` to an SSE stream (one `dupChan` per browser tab) and injects the reload `<script>` into all `text/html` responses:

```haskell
liveReloadMiddleware :: Chan ServerEvent -> Middleware
liveReloadMiddleware chan app req respond
    | requestMethod req == "GET", rawPathInfo req == "/_reload" = do
        chan' <- dupChan chan
        eventSourceAppChan chan' req respond
    | otherwise = app req $ \response -> do
        let ct = maybe "" id $ lookup hContentType (responseHeaders response)
        if "text/html" `BS.isPrefixOf` ct
            then respond (injectReloadScript response)
            else respond response
```

**HTML injection** — `responseToStream` converts `wai-app-static`'s `ResponseFile` (normally a kernel `sendfile`) to a streaming body. Each chunk is converted to a strict `ByteString`, `</body>` is found via `BS8.breakSubstring`, and the script tag is spliced in. `content-length` is stripped from headers since the body size changes:

```haskell
injectReloadScript :: Response -> Response
injectReloadScript response =
    let (status, headers, body) = responseToStream response
        headers' = filter ((/= "content-length") . fst) headers
    in responseStream status headers' $ \send flush ->
        body $ \streamBody -> streamBody (send . patchBuilder) flush
```

**`watch` branch in `main`** — adds `startSiteWatcher` as a third `withAsync` thread alongside the server and Hakyll watcher:

```haskell
("watch":rest) -> do
    let port = case filter (/= "--no-server") rest of { [p] -> read p; _ -> 8000 }
    checkPortFree port
    chan <- newChan
    withAsync (serveWithoutCheck port chan) $ \_ ->
        withAsync (startSiteWatcher chan) $ \_ ->
            withArgs ["watch", "--no-server"] $ hakyll siteRules
```

**Injected script** — stored as a `BS.ByteString` constant `reloadScriptTag`. Saves `window.scrollY` to `sessionStorage` before reload, restores it on `load`:

```javascript
(function(){
  var es = new EventSource('/_reload');
  es.addEventListener('reload', function() {
    sessionStorage.setItem('_lrScroll', window.scrollY);
    window.location.reload();
  });
  window.addEventListener('load', function() {
    var y = sessionStorage.getItem('_lrScroll');
    if (y) { window.scrollTo(0, +y); sessionStorage.removeItem('_lrScroll'); }
  });
})()
```

## Key Decisions

### FSNotify on `_site/` rather than source files
Hakyll's `watch` command rebuilds `_site/` after source changes. Watching `_site/` is the correct signal that a rebuild is *complete* — watching source `.page` files would fire before Hakyll finishes writing, causing the browser to reload a stale or partially-written page.

### Debounce via TVar generation counter
Hakyll writes multiple files per rebuild. Without debouncing, each write would trigger a reload event. The TVar counter pattern is lock-free and handles bursts of any size: each event increments the counter and spawns a short-lived thread; only the thread whose generation is still current after 300ms fires the actual event.

### `dupChan` per SSE connection
`Network.Wai.EventSource.eventSourceAppChan` requires a `Chan ServerEvent`. Using `dupChan` gives each browser tab its own independent read cursor on the broadcast channel, so events reach all tabs and no tab starves another.

### `responseToStream` for HTML injection
`wai-app-static` uses `ResponseFile` (kernel `sendfile`). To inject into the response body, `responseToStream` must be used to convert it to a streaming body — this is the only WAI-level API that handles all response constructors uniformly. The trade-off is that `sendfile` optimization is lost for HTML responses during dev watch. This is acceptable since `NoStore` cache headers already force re-reads and the performance difference is imperceptible locally.

### No template changes
The `<script>` tag is injected at request time by the middleware. The static HTML on disk in `_site/` is never modified. Production builds (`cabal run wiki -- build`) are completely unaffected — no `/_reload` script appears in deployed files.

### browser-sync considered and rejected
`npx browser-sync start --proxy localhost:8000 --files "_site/**/*"` would have required zero code changes but adds an extra port (3000), an extra process, and ~300MB of npm dependencies for a dev-only feature. The integrated approach keeps the workflow at a single `cabal run wiki -- watch` command.

## Important Context for Future Sessions

- **Dev workflow unchanged:** `cabal run wiki -- watch` — now auto-reloads the browser ~1s after each `.page` save. No second terminal or command needed.
- **SSE endpoint:** `http://localhost:8000/_reload` — browsers hold a persistent connection here. Visible in DevTools → Network filtered by "reload".
- **Scroll restoration:** Uses `sessionStorage` keys `_lrScroll`. If scroll restoration seems broken, check that the page's `load` event fires after `scrollTo` (it should, since the script is injected before `</body>`).
- **New cabal deps:** `wai`, `wai-extra`, `fsnotify`, `stm`, `http-types` — all were transitive before; now explicit in `wiki.cabal`.
- **`serveWithoutCheck` signature changed:** now takes `Chan ServerEvent` as second argument. `serveLocally` allocates a fresh `newChan` internally and calls it. Any future callers of `serveWithoutCheck` must supply a channel.
- **Production safety:** `cabal run wiki -- build` (and `build.sh`) never start the server or watcher. The injected script only appears in responses served through the Warp middleware, never in files written to `_site/`.
