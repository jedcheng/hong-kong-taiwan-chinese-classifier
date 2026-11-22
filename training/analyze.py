"""Stage 3. usage: python analyze.py merge NPROC | disp NPROC | final
merge: sum per-file counts -> out/vocab_{words,bigrams,chars}.parquet (+ out/group_sizes.json, out/terms.parquet)
disp : domain dispersion for candidate words (scans DATA/tok) -> out/word_dispersion.parquet
final: stats + filtered lexicons, pair validation, char-variant table -> out/*.csv"""
import sys, os, glob, json, pickle, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATA, WS, VARIANTS
from pairs import PAIRS
import numpy as np, pandas as pd
from multiprocessing import Pool
OUT = f"{WS}/out"
GROUPS = [("hk", "written"), ("hk", "canto"), ("tw", "written"), ("tw", "canto")]
KINDS = {"wdf": "words", "wtf": "words_tf", "bdf": "bigrams", "cdf": "chars", "term": "terms"}

def load(path):
    return pickle.load(open(path, "rb"))

def merge(nproc):
    files = sorted(glob.glob(f"{DATA}/counts/*.pkl")); assert len(files) == int(os.environ.get("NFILES", len(files))), len(files)
    tot = {g: {"n": 0, **{k: collections.Counter() for k in KINDS}} for g in GROUPS}
    with Pool(nproc) as p:
        for c in p.imap_unordered(load, files, chunksize=1):
            for g, v in c.items():
                tot[g]["n"] += v["n"]
                for k in KINDS: tot[g][k].update(v[k])
    json.dump({"_".join(g): tot[g]["n"] for g in GROUPS}, open(f"{OUT}/group_sizes.json", "w"), indent=1)
    for k, name in KINDS.items():
        keys = set().union(*(tot[g][k] for g in GROUPS))
        df = pd.DataFrame({"w": list(keys)})
        for g in GROUPS:
            c = tot[g][k]; df["_".join(g)] = np.array([c.get(w, 0) for w in df.w], dtype=np.int64)
            tot[g][k] = None
        if name != "terms":
            df = df[df[[ "_".join(g) for g in GROUPS]].sum(1) >= 5]
        df.to_parquet(f"{OUT}/vocab_{name}.parquet", index=False)
        print(name, len(df), flush=True)

def stats(df, hk, tw, Nhk, Ntw, a0=10000):
    """DF-based comparison. hk_share = P(HK | word) with both regions given equal weight."""
    y1, y2 = df[hk].astype(float), df[tw].astype(float)
    n1, n2 = y1.sum(), y2.sum()
    a = a0 * (y1 + y2) / (n1 + n2)
    d = np.log((y1 + a) / (n1 + a0 - y1 - a)) - np.log((y2 + a) / (n2 + a0 - y2 - a))
    var = 1 / (y1 + a) + 1 / (y2 + a)
    r1, r2 = y1 / Nhk, y2 / Ntw
    return pd.DataFrame({"w": df.w, "df_hk": y1.astype(int), "df_tw": y2.astype(int),
                         "per10k_hk": r1 * 1e4, "per10k_tw": r2 * 1e4,
                         "hk_share": (r1 / (r1 + r2)).round(4),
                         "log2_ratio": np.log2(((y1 + .5) / Nhk) / ((y2 + .5) / Ntw)).round(3),
                         "z": (d / np.sqrt(var)).round(2)})

def comparisons():
    n = json.load(open(f"{OUT}/group_sizes.json"))
    def cols(df, which):
        if which == "written": return df.assign(HK=df.hk_written, TW=df.tw_written), n["hk_written"], n["tw_written"]
        return df.assign(HK=df.hk_written + df.hk_canto, TW=df.tw_written + df.tw_canto), \
               n["hk_written"] + n["hk_canto"], n["tw_written"] + n["tw_canto"]
    return cols

# ---------------- dispersion ----------------
CAND = None
def init_disp(c):
    global CAND; CAND = c

def disp_file(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path, columns=["domain", "region", "canto", "tokens"])
    acc = collections.defaultdict(collections.Counter)  # (word, region, written?) -> domain counts
    for d, r, cz, toks in zip(*(t.column(i).to_pylist() for i in range(4))):
        wr = cz in ("swc", "neutral")
        for w in set(toks) & CAND:
            acc[(w, r, wr)][d] += 1
    return {k: dict(v) for k, v in acc.items()}

def disp(nproc):
    cols = comparisons()
    v = pd.read_parquet(f"{OUT}/vocab_words.parquet")
    cand = set()
    for which in ("written", "all"):
        df, Nh, Nt = cols(v, which)
        s = stats(df, "HK", "TW", Nh, Nt)
        s = s[(s.df_hk + s.df_tw) >= 30]
        for sign in (1, -1):
            cand |= set(s.loc[(sign * s.z).nlargest(40000).index, "w"])
            ss = s[(s.df_hk + s.df_tw) >= 100]
            cand |= set(ss.loc[(sign * ss.log2_ratio).nlargest(40000).index, "w"])
    print("candidates", len(cand), flush=True)
    files = sorted(glob.glob(f"{DATA}/tok/*.parquet"))
    acc = collections.defaultdict(collections.Counter)
    with Pool(nproc, initializer=init_disp, initargs=(cand,)) as p:
        for r in p.imap_unordered(disp_file, files):
            for k, c in r.items(): acc[k].update(c)
    rows = []
    for w in cand:
        row = {"w": w}
        for reg in ("hk", "tw"):
            for which in ("written", "all"):
                c = acc.get((w, reg, True), collections.Counter()).copy()
                if which == "all": c.update(acc.get((w, reg, False), {}))
                tot = sum(c.values())
                top = c.most_common(1)
                row[f"ndom_{reg}_{which}"] = len(c)
                row[f"topdom_{reg}_{which}"] = top[0][0] if top else ""
                row[f"topshare_{reg}_{which}"] = round(top[0][1] / tot, 3) if top else np.nan
        rows.append(row)
    pd.DataFrame(rows).to_parquet(f"{OUT}/word_dispersion.parquet", index=False)

# ---------------- final ----------------
def final():
    cols = comparisons()
    v = pd.read_parquet(f"{OUT}/vocab_words.parquet")
    dsp = pd.read_parquet(f"{OUT}/word_dispersion.parquet")
    pos = {}
    for fn in ("dict.txt.big", "user_dict.txt"):
        for line in open(f"{WS}/dict/{fn}", encoding="utf-8"):
            x = line.split(); pos.setdefault(x[0], x[2] if len(x) > 2 else "")
    for which in ("written", "all"):
        df, Nh, Nt = cols(v, which)
        s = stats(df, "HK", "TW", Nh, Nt)
        s = s[(s.df_hk + s.df_tw) >= 30].merge(
            dsp[["w"] + [c for c in dsp.columns if c.endswith(which)]].rename(columns=lambda c: c.replace(f"_{which}", "")),
            on="w", how="inner")
        s["pos"] = s.w.map(pos).fillna("oov")  # jieba dict tag: ns place, nr person, nt org, nz other proper, x = HKCanCor
        s["is_cjk"] = s.w.str.contains(r"[㐀-鿿]")
        # favoured side must be spread over >= 10 domains with no domain > 25% of its docs
        hk_side = s.log2_ratio > 0
        s["ndom_fav"] = np.where(hk_side, s.ndom_hk, s.ndom_tw)
        s["topshare_fav"] = np.where(hk_side, s.topshare_hk, s.topshare_tw)
        s["dispersed"] = (s.ndom_fav >= 10) & (s.topshare_fav <= 0.25)
        s = s.sort_values("z", ascending=False)
        s.to_csv(f"{OUT}/lexicon_{which}_full.csv", index=False)
        lex = s[s.dispersed & s.is_cjk]
        lex.to_csv(f"{OUT}/lexicon_{which}.csv", index=False)
        # "distinctive" view: strongly skewed (>= 4x) & reasonably common, ranked by z
        for side, cond, asc in (("hk", lex.log2_ratio >= 2, False), ("tw", lex.log2_ratio <= -2, True)):
            lex[cond & ((lex.df_hk + lex.df_tw) >= 100)].sort_values("z", ascending=asc).head(500) \
                .to_csv(f"{OUT}/top_{side}_{which}.csv", index=False)
        print(which, "N", Nh, Nt, "lexicon", len(lex), flush=True)
    # bigram check (no segmentation): written comparison
    b = pd.read_parquet(f"{OUT}/vocab_bigrams.parquet")
    df, Nh, Nt = cols(b, "written")
    s = stats(df, "HK", "TW", Nh, Nt); s = s[(s.df_hk + s.df_tw) >= 100].sort_values("z", ascending=False)
    s.to_csv(f"{OUT}/bigrams_written.csv", index=False)
    # curated pair validation (raw substring doc frequency)
    t = pd.read_parquet(f"{OUT}/vocab_terms.parquet").set_index("w")
    rows = []
    for which in ("written", "all"):
        df, Nh, Nt = cols(t.reset_index(), which); df = df.set_index("w")
        for cat, a, bb in PAIRS:
            ra, rb = [(df.HK.get(x, 0) / Nh, df.TW.get(x, 0) / Nt) for x in (a, bb)]
            rows.append({"set": which, "category": cat, "hk_term": a, "tw_term": bb,
                         "hk_term_per10k_hk": ra[0] * 1e4, "hk_term_per10k_tw": ra[1] * 1e4,
                         "tw_term_per10k_hk": rb[0] * 1e4, "tw_term_per10k_tw": rb[1] * 1e4,
                         "hk_term_hk_share": ra[0] / (ra[0] + ra[1]) if ra[0] + ra[1] else np.nan,
                         "tw_term_hk_share": rb[0] / (rb[0] + rb[1]) if rb[0] + rb[1] else np.nan,
                         # among docs using either form, share using the HK form, per region
                         "hk_form_pref_in_hk": ra[0] / (ra[0] + rb[0]) if ra[0] + rb[0] else np.nan,
                         "hk_form_pref_in_tw": ra[1] / (ra[1] + rb[1]) if ra[1] + rb[1] else np.nan})
    pd.DataFrame(rows).to_csv(f"{OUT}/pairs_validation.csv", index=False)
    # character variants
    c = pd.read_parquet(f"{OUT}/vocab_chars.parquet").set_index("w")
    df, Nh, Nt = cols(c.reset_index(), "written"); df = df.set_index("w")
    rows = []
    for a, bb in dict.fromkeys(VARIANTS):
        ha, ta, hb, tb = df.HK.get(a, 0) / Nh, df.TW.get(a, 0) / Nt, df.HK.get(bb, 0) / Nh, df.TW.get(bb, 0) / Nt
        rows.append({"hk_form": a, "tw_form": bb, "per10k_hk_hkform": ha * 1e4, "per10k_tw_hkform": ta * 1e4,
                     "per10k_hk_twform": hb * 1e4, "per10k_tw_twform": tb * 1e4,
                     "hkform_pref_in_hk": ha / (ha + hb) if ha + hb else np.nan,
                     "hkform_pref_in_tw": ta / (ta + tb) if ta + tb else np.nan})
    pd.DataFrame(rows).to_csv(f"{OUT}/char_variants.csv", index=False)

if __name__ == "__main__":
    st = sys.argv[1]
    if st == "merge": merge(int(sys.argv[2]))
    elif st == "disp": disp(int(sys.argv[2]))
    elif st == "final": final()
