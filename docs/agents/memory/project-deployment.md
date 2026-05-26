---
name: project-deployment
description: "Wiki is deployed on Hetzner server via nginx, not S3/CloudFlare. Auto-rebuilds every 5 min via cron."
metadata: 
  node_type: memory
  type: project
  originSessionId: bdafa400-5efd-4d20-8aa0-f646fd79b873
---

The wiki moved from S3 + CloudFlare to a Hetzner dedicated server (88.198.62.84, ssh alias `hetzner`) as of May 2025.

**Why:** Self-hosted infrastructure on a beefy dedicated server (i7, 64GB RAM). The S3/CloudFlare deployment scripts (`deploy.sh`, `sync.sh`) are vestigial and unused.

**How to apply:**
- nginx serves `_site/` from `/mnt/sda/private/wiki/` with `Cache-Control: no-cache`
- A cron job (`/etc/cron.d/wiki-rebuild`) runs every 5 min as `deploy` user, pulls both repos, rebuilds if changed
- Server setup docs live in the [sharc repo](https://github.com/shawwn/sharc/blob/main/docs/server-setup.md), not in this repo
- To deploy changes: push to GitHub, wait up to 5 minutes for cron, or SSH in and run the rebuild script manually
- Related project: [[project-sharc-context]]
