---
name: reference-server-access
description: SSH access to the Hetzner server hosting shawwn.net and ycombinator.lol
metadata: 
  node_type: memory
  type: reference
  originSessionId: bdafa400-5efd-4d20-8aa0-f646fd79b873
---

SSH alias `hetzner` connects to 88.198.62.84 as root.

Services run as `deploy` user (uid 1000). Use `sudo` for nginx config and system-level changes.

Key paths:
- Wiki: `/mnt/sda/private/wiki/`
- Wiki-private: `/mnt/sda/private/wiki-private/`
- Haskell env: `source /mnt/sda/private/.ghcup/env`
- nginx config: `/etc/nginx/sites-available/shawwn.net`
- Rebuild script: `/usr/local/bin/wiki-rebuild.sh`
- Rebuild cron: `/etc/cron.d/wiki-rebuild`
- Build log: `/var/log/wiki-build.log`
- sharc (News app): `/opt/sharc/`
