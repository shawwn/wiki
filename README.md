
# Wiki Setup Guide

A Haskell/Hakyll static site generator forked from gwern.net. Content is written in Markdown (`.page` files), compiled to `_site/`, and optionally deployed to S3.

---

## Prerequisites

### macOS

```bash
# System tools
brew install s3cmd tidy-html5 imagemagick exiftool parallel

# Node.js (for MathJax and the webhook server)
brew install node
npm install -g mathjax-node-page

# Haskell toolchain via ghcup
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
# Follow prompts; when asked, let it add ghcup to your shell profile.
source ~/.ghcup/env
```

### Linux (Debian/Ubuntu)

```bash
# Add swap space first if on a small VM — the Haskell build needs ~4 GB RAM
sudo fallocate -l 10G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab
echo 10 | sudo tee /proc/sys/vm/swappiness

# System tools
sudo apt-get update
sudo apt-get install -y build-essential curl libgmp-dev libffi-dev libncurses-dev \
    zlib1g-dev tidy imagemagick parallel s3cmd ripgrep npm

# Node.js MathJax renderer
npm install -g mathjax-node-page

# Haskell toolchain via ghcup
curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org | sh
source ~/.ghcup/env
```

After installation, confirm:

```bash
ghc --version    # should print 9.6.x
cabal --version  # should print 3.x
```

---

## Clone and Build

```bash
git clone https://github.com/shawwn/wiki ~/wiki
cd ~/wiki

# Make ghc/cabal available (add to ~/.bash_profile or ~/.bashrc to make permanent)
source ~/.ghcup/env

# Compile the Haskell generator (~10–20 min on first run; subsequent builds are fast)
cabal build wiki
```

---

## Generate the Site

```bash
# Full pipeline: compile → clean → generate _site/ → sitemap/RSS/MathJax
./build.sh

# Or step by step:
cabal run wiki -- clean   # wipe _site/ and _cache/
cabal run wiki -- build   # generate _site/
```

Output lands in `_site/`.

---

## Serve Locally

Use two terminals for a live development setup:

**Terminal 1** — auto-rebuild on file changes:
```bash
source ~/.ghcup/env
cabal run wiki -- watch --no-server
```

**Terminal 2** — serve `_site/` at http://localhost:8000:
```bash
source ~/.ghcup/env
cabal run wiki -- serve
```

`watch --no-server` watches `.page` files and rebuilds `_site/` incrementally. `serve` is a separate static file server that you leave running; just refresh the browser after each rebuild.

---

## Deploy to S3 (optional)

### One-time S3 setup

1. Create an S3 bucket (e.g. `www.yoursite.com`), uncheck "Block all public access".
2. Enable **Static website hosting** on the bucket.
3. Add a bucket policy allowing public reads:

```json
{
  "Version": "2008-10-17",
  "Statement": [{
    "Sid": "AllowPublicRead",
    "Effect": "Allow",
    "Principal": {"AWS": "*"},
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::www.yoursite.com/*"
  }]
}
```

4. Create an IAM access key and configure s3cmd:

```bash
s3cmd --configure
```

5. Add a DNS CNAME pointing your domain to the S3 website endpoint (e.g. via Cloudflare).

### Configure `env.sh`

Edit `env.sh` and set your bucket, domain name, and RSS metadata:

```bash
export BUCKET="S3://www.yoursite.com"
export WEBSITE="www.yoursite.com"
export NAME="Your Name"
```

### Deploy

```bash
# Build and sync to S3
./deploy.sh

# Or just sync (if _site/ is already built)
./sync.sh
```

---

## Webhook Server (auto-deploy on git push)

`index.js` is an Express server that listens on port 80 and runs `./deploy.sh` when GitHub sends a push webhook.

```bash
npm install        # install Express
npm start          # start server on port 80 (requires root or port forwarding)
```

Configure a GitHub webhook pointing to `http://yourserver/webhooks/github`.

---

## Customizing for a Fork

If you're forking this for your own site, search and replace the existing branding:

```bash
# Find all references to replace
rg -i 'shawn'
rg -i 'shawwn.com'
rg 'UA-'          # Google Analytics tracking IDs

# Files to edit:
# - env.sh          (bucket, domain, name)
# - static/templates/default.html  (site name, analytics, social links)
# - static/templates/analytics.html
# - gen_rss.py      (feed metadata)
```

---

## Directory Structure

```
*.page              Markdown wiki articles
Main.hs             Hakyll site generator
LinkMetadata.hs     Link popup metadata scraper
Inflation.hs        Dollar inflation adjuster
static/
  templates/        HTML templates
  metadata/         Cached link metadata (auto.hs, custom.hs)
  css/ js/ img/     Static assets
docs/               PDFs and other referenced assets
_site/              Generated output (git-ignored)
_cache/             Hakyll build cache (git-ignored)
build.sh            Full build pipeline
deploy.sh           Build + sync to S3
watch.sh            Build + live-reload dev server
env.sh              Site configuration (bucket, domain, RSS)
index.js            GitHub webhook server
```
