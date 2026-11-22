"""Build chunked HK/TW training data from data/extract (written docs, domain-capped, casino-SEO removed).
Split by domain (10% of domains -> test). One random window per doc, length log-uniform 30..1500 chars.
Output: DATA/clf/chunks.parquet  usage: python build_data.py NPROC"""
import sys, os, glob, re, math, random, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, keep_prob, stable_unit
from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq

SPAM = re.compile("娛樂城|百家樂|老虎機|博弈|投注|下注|賠率|賭場|彩票|六合彩|運彩|博彩|真人荷官|出金|賭博|百家乐|娱乐城")
N_TRAIN = {"hk": 450_000, "tw": 450_000}   # per class, approx
DC = None; P = None

def init(dc, p):
    global DC, P; DC, P = dc, p

def window(text, rnd):
    L = int(math.exp(rnd.uniform(math.log(30), math.log(1500))))
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines: return None, L
    i = rnd.randrange(len(lines)); out = lines[i]
    while len(out) < L and i + 1 < len(lines):
        i += 1; out += "\n" + lines[i]
    if len(out) > L:
        s = rnd.randrange(0, len(out) - L + 1) if len(out) > 2 * L else 0
        out = out[s:s + L]
    return out, L

def work(path):
    t = pq.read_table(path, columns=["url", "domain", "region", "canto", "text"])
    rows = collections.defaultdict(list)
    for u, d, r, cz, text in zip(*(t.column(i).to_pylist() for i in range(5))):
        if cz not in ("swc", "neutral"): continue
        if stable_unit(u) >= keep_prob(DC[d]): continue
        if SPAM.search(text): continue
        split = "test" if stable_unit("split:" + d) < 0.10 else "train"
        if stable_unit("take:" + u) >= P[(r, split)]: continue
        rnd = random.Random(u)
        w, L = window(text, rnd)
        if not w or len(w) < 20: continue
        for k, v in zip(("text", "label", "domain", "split", "target_len"), (w, int(r == "hk"), d, split, L)):
            rows[k].append(v)
    return dict(rows)

if __name__ == "__main__":
    files = sorted(glob.glob(f"{DATA}/extract/*/*.parquet")); assert len(files) == int(os.environ.get("NFILES", len(files)))
    dc = collections.Counter()
    for f in files:
        for x in pq.read_table(f, columns=["domain"]).column(0).value_counts().to_pylist(): dc[x["values"]] += x["counts"]
    # expected capped written docs per region (from stage-2 counts): hk 711k, tw 3.89M (before spam removal)
    cap = {"hk": 711_394, "tw": 3_893_566}
    p = {}
    for r in ("hk", "tw"):
        p[(r, "train")] = min(1.0, N_TRAIN[r] / (0.9 * cap[r]) * 1.08)
        p[(r, "test")] = min(1.0, 40_000 / (0.1 * cap[r]))
    print(p, flush=True)
    out = collections.defaultdict(list)
    with Pool(int(sys.argv[1]), initializer=init, initargs=(dict(dc), p)) as pool:
        for r in pool.imap_unordered(work, files):
            for k, v in r.items(): out[k].extend(v)
    os.makedirs(f"{DATA}/clf", exist_ok=True)
    tbl = pa.table(dict(out)); pq.write_table(tbl, f"{DATA}/clf/chunks.parquet", compression="zstd")
    df = tbl.select(["label", "split"]).to_pandas()
    print(df.groupby(["split", "label"]).size(), flush=True)
