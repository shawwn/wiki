# Handoff: README Rewrite and cabal v2- Cleanup

**Date:** 2026-04-16  
**Session:** 2026-04-16-005-readme-and-cabal-cleanup

## What Was Accomplished

1. Rewrote `README.md` from scratch with complete setup instructions for macOS and Linux, covering prerequisites, clone + build, local serving, S3 deployment, the webhook server, and fork customization.
2. Replaced all `cabal v2-build` / `cabal v2-run` invocations with `cabal build` / `cabal run` in `README.md`, `build.sh`, and `watch.sh`.

### Files Changed

- **`README.md`** — full rewrite; structured sections replacing loose notes
- **`build.sh`** — `cabal v2-build wiki` → `cabal build wiki`, `cabal v2-run` → `cabal run`
- **`watch.sh`** — `cabal v2-run` → `cabal run`

## Key Decisions

### Why v2- was removed
`cabal v2-*` commands are aliases for the now-default Nix-style build system introduced in cabal 3.x. `cabal run`, `cabal build`, etc. are identical and the canonical form. The `v2-` prefix is legacy and just adds noise.

### README scope
The old README was a mix of setup notes, one-off VM instructions, and historical cabal init notes. The rewrite keeps only what's needed to go from zero to a running site: prerequisites, build, serve, optional S3 deploy, and fork customization pointers. Historical/internal notes (cabal init steps, gwern.net fork VM setup) were dropped.

## Important Context for Future Sessions

- **`build.sh`** calls `cabal build wiki` then `cabal run wiki -- clean` then `cabal run wiki -- build` — the explicit `cabal build` step before `run` is intentional (compiles with optimizations before generating).
- **`watch.sh`** runs the full `build.sh` first, then starts `cabal run wiki -- watch --no-server` in the background, then uses `entr` to re-run `build-mathjax.sh` on changes.
- **Dev server:** `cabal run wiki -- watch` or `cabal run wiki -- serve`; default port 8000.
- **S3 deploy:** `./deploy.sh` (build + sync) or `./sync.sh` (sync only). Requires `s3cmd --configure` and `env.sh` edited with bucket/domain.
