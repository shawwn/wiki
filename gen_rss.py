# sudo apt-get install python3-pip
# sudo pip3 install rfeed
# sudo npm i -g html-beautify
# python3 gen_rss.py | html-beautify > feed.rss

from glob import glob
from datetime import datetime
from os import environ as env

RSS_AUTHOR=env['RSS_AUTHOR'] if 'RSS_AUTHOR' in env else "Gwern Branwen"
RSS_URL=env['RSS_URL'] if 'RSS_URL' in env else "https://www.gwern.net/"
RSS_TITLE=env['RSS_TITLE'] if 'RSS_TITLE' in env else "Essays - Gwern.net"
RSS_DESC=env['RSS_DESC'] if 'RSS_DESC' in env else "This is the website of Gwern Branwen, writer & independent researcher. I am most interested in psychology, statistics, and technology; I am best known for my writings on the darknet markets & Bitcoin, blinded self-experiments & Quantified Self analyses, dual n-back & spaced repetition, and modafinil."

files = []

def kv(s):
    xs = s.split(': ', 1)
    k, v = xs if len(xs) > 1 else (xs[0], '')
    return [k, v]

for name in glob('*.page'):
    s = open(name).read()
    if not '\n...\n' in s:
        continue
    header, body = s.split('\n...\n', 1)
    if header.startswith('---\n'):
        header = header[3:]
    h = dict([kv(line) for line in header.strip().splitlines()])
    h['body'] = body
    if 'created' in h:
        d, m, y = h['created'].split()
        m = m[:3]
        dt = datetime.strptime(' '.join([d, m, y]), '%d %b %Y')
        ds = dt.strftime("%a, %d %b %Y %H:%M:%S %z")
        sortby = dt.strftime("%Y %m %d")
        h['created'] = ds
        h['sortby'] = sortby
    files += [[name, h]]

import json
from rfeed import *
items = []

import os
exclude = []
if os.path.isfile('exclude.txt'):
    exclude = open('exclude.txt').read().splitlines()

files = sorted(files, key=lambda x: x[1]['sortby'], reverse=True)
for name, h in files:
    base = name.split('.page')[0].strip()
    if base in exclude or name in exclude:
        continue
    title = h['title'] if 'title' in h else base
    url = RSS_URL + base
    desc = h['description'] if 'description' in h else ''
    item = Item(
            title=title,
            link=url,
            description=desc,
            author=RSS_AUTHOR,
            pubDate=datetime.strptime(h['sortby'], '%Y %m %d')
            )
    items += [item]

feed = Feed(
        title=RSS_TITLE,
        description=RSS_DESC,
        link=RSS_URL,
        language="en-US",
        items=items)

print(feed.rss())


