# Handoff: Dependabot Express Security Upgrade

**Date:** 2026-04-25  
**Session:** 2026-04-25-003-dependabot-express-upgrade

## What Was Accomplished

Resolved all 16 open Dependabot security alerts (5 high, 6 medium, 5 low) by upgrading `express` from `4.17.1` to `4.22.1`. All vulnerable packages were transitive dependencies of express; a single version bump fixed everything.

### Files Changed

- **`package.json`** — `express` version range changed from `^4.17.1` to `^4.22.1`
- **`yarn.lock`** — regenerated; all transitive deps updated to fixed versions

### Alerts Resolved

| # | Severity | Package | Fixed by |
|---|----------|---------|---------|
| #23, #20, #13 | High | `path-to-regexp` ReDoS (3 variants) | `path-to-regexp@0.1.12` |
| #14 | High | `body-parser` DoS (url encoding) | `body-parser@1.20.3` |
| #1 | High | `qs` Prototype Pollution | `qs@6.14.0` |
| #21, #22 | Medium | `qs` DoS (bracket/comma parsing) | `qs@6.14.0` |
| #5, #6, #11, #19 | Medium | `jquery` XSS (3 variants) | updated via express |
| #12 | Medium | `express` Open Redirect | `express@4.22.1` |
| #15 | Low | `express` XSS via redirect | `express@4.22.1` |
| #16, #17 | Low | `serve-static`/`send` template injection XSS | updated via express |
| #18 | Low | `cookie` out-of-bounds characters | updated via express |

### PR Actions

- Closed Dependabot PR #6 (`Bump express from 4.17.1 to 4.17.3`) — superseded; that bump would not have fixed the high-severity CVEs.

## Key Decisions

### Single express upgrade rather than merging Dependabot PRs
Dependabot had only opened one PR (#6, proposing `4.17.3`) which would not have fixed the high-severity path-to-regexp, body-parser, or qs issues. Upgrading directly to `4.22.1` (latest stable 4.x) resolved all 16 alerts at once. Express 4.22.1 pins `path-to-regexp@~0.1.12`, `body-parser@~1.20.3`, `qs@~6.14.0`.

### yarn over npm
The repo uses `yarn.lock` (yarn 1.x). Used `yarn install` after editing `package.json` to regenerate the lockfile cleanly.

## Important Context for Future Sessions

- **No open Dependabot alerts remain** as of 2026-04-25. `gh api repos/shawwn/wiki/dependabot/alerts?state=open` returns `[]`.
- **The Node.js code is only a webhook server** (`index.js`, Express on port 80) that triggers `./deploy.sh` on GitHub push events. It is not part of the Haskell build pipeline or the generated site.
- **jquery** appears in the alert list but is not in `package.json` or `node_modules` directly — it was flagged as a transitive dep that got resolved by the express upgrade.
- **Commit:** `72abce3` — "Upgrade express 4.17.1 → 4.22.1 to fix high-severity CVEs"
