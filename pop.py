import sys
import subprocess

def _stdin(stdin):
    if type(stdin) in [str, bytes]:
        return subprocess.PIPE, stdin
    else:
        return stdin, None

def _popen(cmd, stdout, stdin, shell):
    stdin, input = _stdin(stdin)
    p = subprocess.Popen(cmd, stdout=stdout, stdin=stdin, shell=shell)
    out, err = p.communicate(input=input)
    return out, err

def _system(cmd, args, stdin=None, stdout=subprocess.PIPE):
    out, err = _popen([cmd] + args, stdout=stdout, stdin=stdin, shell=False)
    return out, err

def system(cmd, args, stdin=None, stdout=subprocess.PIPE):
    out, err = _system(cmd, args, stdin=stdin, stdout=stdout)
    return out


def kv(s):
  xs = s.split(': ', 1)
  k, v = xs if len(xs) > 1 else (xs[0], '')
  return (k, v)

def md(s):
  return system('pandoc', ['--from=markdown', '--write=html'], stdin=s.encode()).decode('utf-8')

def esc(s):
  r = ""
  for c in s:
    if c == '\\':
      r += '\\\\'
    elif c == '"':
      r += '\\"'
    elif c == '\n':
      r += '\\n'
    else:
      r += c
  return '"' + r + '"'

xs = [x for x in ('\n' + sys.stdin.read()).split('\n---') if len(x.strip()) > 0]
print('[')
n = len(xs)
for i in range(n):
  x = xs[i]
  header, body = x.split('...')
  h = dict([kv(x) for x in header.strip().splitlines()])
  body = body.lstrip()
  html = md(body)
  #print(repr(header))
  #print(md(body))
  print('(' + esc(h['url'] if 'url' in h else ''))
  print(' , ( ' + esc(h['title'] if 'title' in h else ''))
  print('   , ' + esc(h['author'] if 'author' in h else ''))
  print('   , ' + esc(h['date'] if 'date' in h else ''))
  print('   , ' + esc(h['doi'] if 'doi' in h else ''))
  print('   , ' + esc(html) + ')')
  print(' ),' if i < n - 1 else ' )')
print(']')
