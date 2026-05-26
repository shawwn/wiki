---
name: project-sharc-context
description: The sharc repo is an Arc-on-Common-Lisp runtime that runs on the same Hetzner server
metadata: 
  node_type: memory
  type: reference
  originSessionId: bdafa400-5efd-4d20-8aa0-f646fd79b873
---

[sharc](https://github.com/shawwn/sharc) is an independent open-source Arc-on-Common-Lisp runtime (like HN's Clarc). It runs the News app at news.ycombinator.lol on the same Hetzner server.

The [server-setup.md](https://github.com/shawwn/sharc/blob/main/docs/server-setup.md) in that repo is the canonical reference for the Hetzner server's disk layout, nginx vhosts, TLS certs, cron jobs, and user accounts. It covers both the wiki and the News app.

Related: [[project-deployment]], [[reference-server-access]]
