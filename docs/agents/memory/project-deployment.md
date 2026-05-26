---
name: project-deployment
description: "Wiki is deployed on Hetzner server via nginx, not S3/CloudFlare. Auto-rebuilds every 5 min via cron."
metadata: 
  node_type: memory
  type: project
  originSessionId: bdafa400-5efd-4d20-8aa0-f646fd79b873
---

The wiki moved from S3 + CloudFlare to a Hetzner dedicated server (88.198.62.84, ssh alias `hetzner`) as of May 2025.

**Why:** Self-hosted infrastructure on a beefy dedicated server (i7, 64GB RAM). The old S3 scripts (`sync.sh`, `predeploy.sh`) are kept as reference but the active flow is `deploy.sh`.

**How to apply:**
- `./deploy.sh` is the primary deploy command: runs predeploy locally, pushes both repos, SSHs into hetzner, pulls both, re-overlays, and runs `./build.sh`
- nginx serves `_site/` from `/mnt/sda/private/wiki/` with `Cache-Control: no-cache`
- A cron job (`/etc/cron.d/wiki-rebuild`) also runs every 5 min as `deploy` user as a fallback
- Server setup docs live in the [sharc repo](https://github.com/shawwn/sharc/blob/main/docs/server-setup.md), not in this repo
- git safe.directory is configured on the server for both `/mnt/sda/private/wiki` and `/mnt/sda/private/wiki-private` (needed because SSH is root but repos are owned by deploy)
- Related project: [[project-sharc-context]]
