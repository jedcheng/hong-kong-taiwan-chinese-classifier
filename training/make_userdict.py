"""Build dict/user_dict.txt: variant spellings of dict.txt.big words + HKCanCor Cantonese words."""
import sys, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import WS, VARIANTS
words = {}
for line in open(f"{WS}/dict/dict.txt.big", encoding="utf-8"):
    w, f, *t = line.split()
    words[w] = (int(f), t[0] if t else "n")
vmap = {}
for a, b in VARIANTS:
    vmap.setdefault(a, b); vmap.setdefault(b, a)
add = {}
for w, (f, t) in words.items():
    if len(w) < 2 or not any(c in vmap for c in w): continue
    v = "".join(vmap.get(c, c) for c in w)
    if v != w and v not in words:
        add[v] = (f, t)
n_var = len(add)
import pycantonese
c = collections.Counter(tok.word for tok in pycantonese.hkcancor().tokens())
n_can = 0
for w, f in c.items():
    if len(w) >= 2 and all('一' <= ch <= '鿿' for ch in w) and w not in words and w not in add:
        add[w] = (max(3, f), "x"); n_can += 1
with open(f"{WS}/dict/user_dict.txt", "w", encoding="utf-8") as fo:
    for w, (f, t) in add.items():
        fo.write(f"{w} {f} {t}\n")
print("variant entries", n_var, "hkcancor entries", n_can)
