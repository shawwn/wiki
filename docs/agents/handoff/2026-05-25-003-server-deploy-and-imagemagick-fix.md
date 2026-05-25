# Handoff: Server deploy and ImageMagick fix (2026-05-25)

**Date:** 2026-05-25
**Session:** 2026-05-25-003-server-deploy-and-imagemagick-fix

## What Was Accomplished

### Wiki deployed to Hetzner server

The wiki is now building and serving from a Hetzner dedicated server
(88.198.62.84, hostname `ln1`). Full details of the server setup are
in the sharc7 repo (`docs/server-setup.md`), but the wiki-specific
parts are:

- GHC 9.6.6 + cabal 3.14 installed on HDD (`/mnt/sda/private/.ghcup/`,
  `/mnt/sda/private/cabal/`)
- Wiki cloned to `/mnt/sda/private/wiki/`
- wiki-private cloned to `/mnt/sda/private/wiki-private/`
- Built site at `/mnt/sda/private/wiki/_site/` (nginx will serve this)
- Auto-rebuild cron every 5 minutes (`/usr/local/bin/wiki-rebuild.sh`)
  checks both repos, pulls if changed, runs overlay, rebuilds

### Branch renamed master -> main

Used `gh api repos/shawwn/wiki/branches/master/rename -f new_name=main`
to rename the default branch on GitHub. All clones (local and server)
updated.

### wiki-private repo created on GitHub

`github.com/shawwn/wiki-private` (private repo) initialized with the
test content from session 002 and pushed. Cloned to server as a
sibling of the wiki repo.

### ImageMagick remote URL fix

**Commit:** `a8a5556` - Skip remote URLs in staticImg

`staticImg` in Main.hs was running ImageMagick's `identify` on remote
image URLs (e.g. `https://i.imgur.com/...`). `identify` can't fetch
HTTP, so it returned ExitFailure, which called `error`, which Hakyll
caught and wrote a 0-byte output file. This caused memorybox.page
and swarm.page to produce empty build outputs.

Fix: added `http://` and `https://` prefix checks to the `vector`
guard in `staticImg`, so remote URLs are returned unchanged (same as
SVGs and data URIs).

### Files Changed

- **`Main.hs`** - Added http/https URL skip in `staticImg` (line 350)

## Key Decisions

### TMPDIR must be on the HDD

The server sets `TMPDIR=/mnt/sda/private/wiki/tmp` because
LinkMetadata.hs uses `renameFile` from `/tmp` to
`static/metadata/auto.hs`. Since `/tmp` is on the SSD and the wiki
is on the HDD, `rename` fails with "Invalid cross-device link".
This is configured in `/etc/profile.d/haskell.sh` on the server.

### Cron polling over webhooks

5-minute `git fetch` + diff check is simpler than a webhook server.
No extra port, no GitHub webhook config, no failure modes to manage.

## Important Context for Future Sessions

### Server paths

| Path | What |
|---|---|
| `/mnt/sda/private/wiki/` | wiki repo clone |
| `/mnt/sda/private/wiki-private/` | wiki-private repo clone |
| `/mnt/sda/private/wiki/_site/` | built site (nginx serves this) |
| `/mnt/sda/private/wiki/tmp/` | TMPDIR for builds |
| `/mnt/sda/private/.ghcup/` | GHC/cabal (2.9GB) |
| `/mnt/sda/private/cabal/` | cabal package store (2.2GB) |
| `/usr/local/bin/wiki-rebuild.sh` | auto-rebuild script |
| `/etc/profile.d/haskell.sh` | env vars for Haskell builds |

### SSH access

```
ssh hetzner   # alias for root@88.198.62.84 (1Password SSH agent)
```

Services run as `deploy` user (uid 1000), not root.

### Remaining build warnings

- Some non-fatal `[ERROR]` lines from LinkMetadata.hs when it fails
  to fetch external URLs for metadata. These don't block the build
  after the ImageMagick fix, but the metadata for those links will
  be missing.

### Still TODO (tracked in sharc7 repo)

- Nginx vhost for shawwn.net (serve _site/)
- DNS records in Route 53
- TLS via Let's Encrypt
