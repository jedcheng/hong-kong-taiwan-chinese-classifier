"""Stage 2: domain-capped sample (<= CAP expected docs per registered domain, md5(url) sampling),
jieba-tokenize, and count per group (region x written/cantonese).
Writes DATA/tok/<name>.parquet (tokens per sampled doc) and DATA/counts/<name>.pkl.
usage: python tokenize_count.py NPROC"""
import sys, os, glob, re, pickle, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, keep_prob, stable_unit, init_jieba
from pairs import PAIRS
from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

CJK = re.compile(r"[㐀-鿿豈-﫿]")
KEEP = re.compile(r"[㐀-鿿豈-﫿A-Za-z]")
CJKRUN = re.compile(r"[㐀-鿿豈-﫿]+")
TERMS = sorted({t for _, a, b in PAIRS for t in (a, b)})
DCOUNT = None

def init(dc):
    global DCOUNT, jieba
    DCOUNT = dc; jieba = init_jieba()

def group(region, canto):
    return (region, "written" if canto in ("swc", "neutral") else "canto")

def work(path):
    name = os.path.basename(path).replace(".parquet", "")
    out_tok, out_cnt = f"{DATA}/tok/{name}.parquet", f"{DATA}/counts/{name}.pkl"
    if os.path.exists(out_cnt): return name, -1
    C = collections.defaultdict(lambda: {"n": 0, "wdf": collections.Counter(), "wtf": collections.Counter(),
                                         "bdf": collections.Counter(), "cdf": collections.Counter(),
                                         "term": collections.Counter()})
    rows = {k: [] for k in ["url", "dump", "domain", "region", "canto", "tokens"]}
    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=5000, columns=["url", "dump", "domain", "region", "canto", "text"]):
        for u, dump, d, r, cz, t in zip(*(b.column(i).to_pylist() for i in range(6))):
            if stable_unit(u) >= keep_prob(DCOUNT[d]): continue
            toks = [w.lower() for w in (x.strip() for x in jieba.lcut(t)) if w and KEEP.search(w)]
            g = C[group(r, cz)]
            g["n"] += 1
            g["wtf"].update(toks); g["wdf"].update(set(toks))
            bg = set()
            for run in CJKRUN.findall(t):
                bg.update(run[i:i+2] for i in range(len(run) - 1))
            g["bdf"].update(bg)
            g["cdf"].update(set(CJK.findall(t)))
            g["term"].update(x for x in TERMS if x in t)
            for k, v in zip(rows, (u, dump, d, r, cz, toks)): rows[k].append(v)
    os.makedirs(f"{DATA}/tok", exist_ok=True); os.makedirs(f"{DATA}/counts", exist_ok=True)
    pq.write_table(pa.table(rows), out_tok, compression="zstd")
    with open(out_cnt + ".tmp", "wb") as f:
        pickle.dump({k: {kk: (dict(vv) if kk != "n" else vv) for kk, vv in v.items()} for k, v in C.items()}, f)
    os.replace(out_cnt + ".tmp", out_cnt)
    return name, len(rows["url"])

if __name__ == "__main__":
    files = sorted(glob.glob(f"{DATA}/extract/*/*.parquet"))
    assert len(files) == int(os.environ.get("NFILES", len(files))), len(files)
    dc = collections.Counter()
    for f in files:
        v = pq.read_table(f, columns=["domain"]).column(0).value_counts()
        for x in v.to_pylist(): dc[x["values"]] += x["counts"]
    print("domains", len(dc), "docs", sum(dc.values()), flush=True)
    with Pool(int(sys.argv[1]), initializer=init, initargs=(dict(dc),)) as p:
        for name, n in p.imap_unordered(work, files):
            print(n, name, flush=True)
