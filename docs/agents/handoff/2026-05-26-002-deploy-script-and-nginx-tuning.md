# Handoff: Deploy Script and Nginx Tuning

**Date:** 2026-05-26
**Session:** 2026-05-26-002-deploy-script-and-nginx-tuning

## What Was Accomplished

### Nginx caching disabled

Added `Cache-Control: no-cache` to the nginx config for shawwn.net.
The server is a dedicated i7 with 64GB RAM; there is no reason to
cache. A backup of the previous config was saved to
`/etc/nginx/sites-available/shawwn.net.bak`.

### deploy.sh rewritten for Hetzner

Replaced the old S3-based deploy.sh with a script that:
1. Runs `./predeploy.sh` (unlinks overlay, resets build artifacts)
2. Pushes both `wiki` and `wiki-private` to GitHub
3. SSHs into hetzner, pulls both repos, re-overlays, runs `./build.sh`

The old S3 flow is preserved as a comment at the bottom. `predeploy.sh`
and `sync.sh` are untouched.

### CLAUDE.md updated

Deployment section rewritten to reflect Hetzner/nginx (was still
referencing S3, CloudFlare, Express webhook server). Added content
notes about the hand-written `index` file and the duplicated sidebar.

### Nginx: docs/ autoindex and .md as plain text

- Added `autoindex on` for `/docs/` so directory listings work
- Added a location block serving `.md` files as `text/plain; charset=utf-8`
  (they were being served as `text/html` due to `default_type`, rendering
  as an unreadable wall of text)

### Git safe.directory fix

`deploy.sh` SSHs as root but the repos are owned by `deploy`. Added
both wiki directories to root's global git safe.directory config:
```
git config --global --add safe.directory /mnt/sda/private/wiki
git config --global --add safe.directory /mnt/sda/private/wiki-private
```

### Agent memory files added to repo

Created `docs/agents/memory/` with project context files (deployment
setup, server access, sharc context). Symlinked from Claude's memory
path so the memory system picks them up. These are version-controlled
alongside the handoff files.

## Files Changed

- **`CLAUDE.md`** -- deployment section rewritten, content notes added
- **`deploy.sh`** -- rewritten for Hetzner (push, SSH, pull, build)
- **`docs/agents/handoff/2026-05-26-001-nginx-caching-and-docs-update.md`** -- handoff for the earlier CLAUDE.md update
- **`docs/agents/memory/`** -- new directory with MEMORY.md index and 3 memory files

## Server Changes (not in repo)

- `/etc/nginx/sites-available/shawwn.net` -- Cache-Control: no-cache, autoindex for /docs/, text/plain for .md
- Root's `~/.gitconfig` -- safe.directory entries for both wiki repos

## Important Context for Future Sessions

- The current nginx config on the server is ahead of what is tracked
  in the repo (nginx configs are not version-controlled in this repo;
  they are documented in the sharc repo's server-setup.md).
- The sidebar nav links are duplicated in `static/templates/default.html`
  and `index`. Any sidebar change needs both files.
- The `index` file is hand-written HTML, not generated from a `.page`
  file. Hakyll copies it as-is.
