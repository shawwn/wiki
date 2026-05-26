# Handoff: Nginx Caching Fix and CLAUDE.md Update

**Date:** 2026-05-26
**Session:** 2026-05-26-001-nginx-caching-and-docs-update

## What Was Accomplished

### Nginx caching fix

After deploying content changes, shawwn.net was still showing old
content. The nginx config at `/etc/nginx/sites-available/shawwn.net`
had no `Cache-Control` headers, so browsers cached aggressively based
on heuristic expiration from `Last-Modified`/`ETag`.

Fix: added `add_header Cache-Control "no-cache"` to the main server
block. This tells browsers to revalidate with the server on every
request (they still use conditional GETs with `If-Modified-Since`, so
unchanged files return 304). No per-path caching exceptions; the
server is a dedicated i7 with 64GB RAM, so there is no reason to cache.

A backup of the previous config was saved to
`/etc/nginx/sites-available/shawwn.net.bak`.

### CLAUDE.md updated

The deployment section was still referencing S3, CloudFlare, the
Express.js webhook server, and `s3cmd`. Updated to reflect the current
Hetzner/nginx setup:

- Serving via nginx from `_site/`
- Auto-rebuild cron every 5 minutes
- Private content overlay via `overlay.sh`
- Removed Node.js from key dependencies
- Documented that the `index` file is hand-written HTML (not a `.page`)
- Documented that the sidebar is duplicated in `default.html` and `index`

## Important Context for Future Sessions

- The nginx config lives at `/etc/nginx/sites-available/shawwn.net` on
  the server (ssh alias `hetzner`). Test with `sudo nginx -t`, apply
  with `sudo systemctl reload nginx`.
- Server setup details (disk layout, users, cron, nginx vhosts, TLS)
  are tracked in [sharc/docs/server-setup.md](https://github.com/shawwn/sharc/blob/main/docs/server-setup.md), not in this repo.
- The `deploy.sh` / `sync.sh` / `predeploy.sh` scripts in this repo
  still reference the old S3 workflow. They are not used on the server;
  the cron job at `/etc/cron.d/wiki-rebuild` handles deployment.
