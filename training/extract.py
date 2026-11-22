"""Stage 1: keep docs whose hostname TLD is .hk or .tw; add host/domain/region/dump and a
CantoneseDetect label. One output parquet per input file in DATA/extract/<dump>/.
usage: python extract.py NPROC"""
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SRC, DATA, host_info
from multiprocessing import Pool
import pyarrow as pa, pyarrow.parquet as pq

def work(path):
    dump = os.path.basename(os.path.dirname(path))
    out = f"{DATA}/extract/{dump}/{os.path.basename(path)}"
    if os.path.exists(out): return out, -1
    from cantonesedetect import CantoneseDetector
    det = CantoneseDetector(split_seg=False, use_quotes=False)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    keys = ["url", "timestamp", "dump", "host", "domain", "region", "canto", "text"]
    w, n = None, 0
    for b in pq.ParquetFile(path).iter_batches(batch_size=20000, columns=["url", "timestamp", "text"]):
        cols = {k: [] for k in keys}
        for u, ts, t in zip(*(b.column(i).to_pylist() for i in range(3))):
            h, d, r = host_info(u)
            if not r: continue
            for k, v in zip(keys, (u, str(ts), dump, h, d, r, str(det.judge(t)), t)): cols[k].append(v)
        tbl = pa.table(cols, schema=pa.schema([(k, pa.string()) for k in keys]))
        if w is None: w = pq.ParquetWriter(out + ".tmp", tbl.schema, compression="zstd")
        w.write_table(tbl); n += len(tbl)
    w.close()
    os.replace(out + ".tmp", out)
    return out, n

if __name__ == "__main__":
    files = sorted(glob.glob(f"{SRC}/*/*.parquet"))
    with Pool(int(sys.argv[1])) as p:
        for out, n in p.imap_unordered(work, files):
            print(n, out, flush=True)
