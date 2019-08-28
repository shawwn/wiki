
## Running on macOS

```
brew install s3cmd
brew install tidy-html5
brew install imagemagick
npm i -g mathjax-node-page

#
# Setting up Haskell
#

# https://github.com/haskell/ghcup

# complete bootstrap
curl https://gitlab.haskell.org/haskell/ghcup/raw/master/bootstrap-haskell -sSf | sh

# prepare your environment
. "$HOME/.ghcup/env"
echo '. $HOME/.ghcup/env' >> "$HOME/.bash_profile"

#
# Setting up the website
#

git clone https://github.com/shawwn/wiki ~/wiki
cd ~/wiki

# edit env.sh and fill in your own name, website URL, and S3 bucket

# build the project
cabal v2-build

# you can generate the site with this, or by running ./build.sh (see Deployment section below)
cabal v2-run wiki -- build

```

## Notes on setting up a fork of gwern.net

```
#
# Setting up the VM
#

# create a Digital Ocean Ubuntu droplet (the $5/mo plan is fine) and ssh into it

# install tmux
apt-get install tmux
tmux new -s wiki

# add 10GB of swap space
sudo fallocate -l 10G /swapfile
sudo dd if=/dev/zero of=/swapfile bs=10240 count=1048576
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
sudo echo '/swapfile swap swap defaults 0 0' >> /etc/fstab
sudo free -h
echo 10 | sudo tee /proc/sys/vm/swappiness
echo vm.swappiness = 10 | sudo tee -a /etc/sysctl.conf

#
# Setting up Haskell
#

# install haskell prereqs
sudo apt-get update
sudo apt-get install build-essential curl libgmp-dev libffi-dev libncurses-dev -y

# https://github.com/haskell/ghcup

# complete bootstrap
curl https://gitlab.haskell.org/haskell/ghcup/raw/master/bootstrap-haskell -sSf | sh

# prepare your environment
. "$HOME/.ghcup/env"
echo '. $HOME/.ghcup/env' >> "$HOME/.bashrc" # or similar


#
# Setting up the website
#

git clone https://github.com/shawwn/wiki ~/wiki
cd ~/wiki

# edit env.sh and fill in your own name, website URL, and S3 bucket

# zlib headers are required
sudo apt-get install zlib1g-dev

# build the project
cabal v2-build

# you can generate the site with this, or by running ./build.sh (see Deployment section below)
cabal v2-run wiki -- build
```

## Notes on what I did to create wiki.cabal (skip this section)
```
cabal init -n --is-executable
cabal v2-run

echo dist-newstyle >> .gitignore
git add .gitignore
git commit -m "cabal init -n --is-executable && cabal v2-run"

# required packages:
# pandoc missingh happy pretty-show tagsoup arxiv aeson hakyll

# Add the following to wiki.cabal:
# base >=4.12 && <4.13, bytestring >=0.10 && <0.11, containers >=0.6 && <0.7, text >=1.2 && <1.3, directory >=1.3 && <1.4, pandoc >=2.7.2 && <= 2.7.3, MissingH ==1.4.1.0, pretty-show ==1.9.5, aeson ==1.4.2.0, tagsoup == 0.14.7, arxiv == 0.0.1, hakyll == 4.12.5.2, filestore ==0.6.3.4
```

## Notes on deployment
```
# install ripgrep (optional)
snap install ripgrep --classic

# install required prerequisites
sudo apt-get install parallel
sudo apt-get install npm
npm i -g mathjax-node-page
sudo apt-get install tidy
sudo apt-get install imagemagick

# - create s3 bucket. I used US East (Ohio)
# - uncheck "Block all public access"

# - go to top right -> "My Security Credentials" -> Access Keys (access key ID and secret access key)
# - create a new access key

# install s3cmd
sudo apt-get install s3cmd

# https://kunallillaney.github.io/s3cmd-tutorial/
s3cmd --configure

# paste your access key and secret key

# add public buket policy

{"Version": "2008-10-17",
"Statement": [{"Sid": "AllowPublicRead",
"Effect": "Allow",
"Principal": {
"AWS": "*"
},
"Action": "s3:GetObject",
"Resource": "arn:aws:s3:::www.shawwn.com/*"
}]}

# Turn on S3 static site hosting on your bucket
# https://docs.aws.amazon.com/AmazonS3/latest/user-guide/static-website-hosting.html

# Add a cloudflare CNAME entry to your bucket
# CNAME www www.shawwn.com.s3-website.us-east-2.amazonaws.com 

# Update all pages with your own name and url
rg 'Shawn' -i

# Create a google analytics account

# Change google analytics codes. Search for "UA-" and replace with your own code
rg UA-

# Sign up at https://tinyletter.com/

# Create a subreddit

# Create a patreon

# Create a disqus account

# Run ./sync.sh to build and deploy your site
```
