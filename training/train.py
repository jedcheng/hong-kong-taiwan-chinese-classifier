"""Train char-ngram logistic regression (full + style), evaluate by length, save models.
usage: python train.py NJOBS"""
import sys, os, time, json, numpy as np, pandas as pd, joblib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from hkness import make_vectorizer, MASK_TOKEN, DIGIT_RE, WS_RE
from common import DATA as _D, WS
import re
from mask_terms import HK_PLACES, TW_PLACES
MASK_RE = re.compile("|".join(sorted(map(re.escape, HK_PLACES + TW_PLACES), key=len, reverse=True)))
def normalize(text, mask=True):
    t = WS_RE.sub(" ", DIGIT_RE.sub("0", text.lower()))
    return MASK_RE.sub(MASK_TOKEN, t) if mask else t
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import roc_auc_score, accuracy_score, log_loss
from joblib import Parallel, delayed
import scipy.sparse as sp
DATA = f"{_D}/clf"
OUT = f"{WS}/models"; os.makedirs(OUT, exist_ok=True)
NJ = int(sys.argv[1])
df = pd.read_parquet(f"{DATA}/chunks.parquet")
vec = make_vectorizer()

def featurize(texts, mask):
    parts = Parallel(n_jobs=NJ)(delayed(lambda ts: vec.transform([normalize(t, mask) for t in ts]))(texts[i:i + 20000])
                                for i in range(0, len(texts), 20000))
    return sp.vstack(parts).tocsr()

tr, te = df[df.split == "train"], df[df.split == "test"].copy()
# balance train classes exactly
n = tr.label.value_counts().min()
tr = pd.concat([g.sample(n, random_state=0) for _, g in tr.groupby("label")]).sample(frac=1, random_state=1)
# balance test classes too, so accuracy is meaningful
m = te.label.value_counts().min()
te = pd.concat([g.sample(m, random_state=0) for _, g in te.groupby("label")])
te["len"] = te.text.str.len()
bins = [0, 50, 100, 200, 500, 2000]
te["len_bin"] = pd.cut(te.len, bins).astype(str)
print("train", len(tr), "test", len(te), flush=True)
report = {}
for name, mask in (("style", True), ("full", False)):
    t0 = time.time()
    Xtr = featurize(tr.text.tolist(), mask); Xte = featurize(te.text.tolist(), mask)
    print(name, "featurize", round(time.time() - t0), "s nnz", Xtr.nnz, flush=True)
    best = None
    for alpha in (1e-7, 3e-7, 1e-6, 3e-6):
        t0 = time.time()
        clf = SGDClassifier(loss="log_loss", alpha=alpha, max_iter=15, tol=1e-4, random_state=0)
        clf.fit(Xtr, tr.label.values)
        p = clf.predict_proba(Xte)[:, 1]
        r = {"auc": roc_auc_score(te.label, p), "acc": accuracy_score(te.label, p > .5), "logloss": log_loss(te.label, p)}
        print(name, "alpha", alpha, {k: round(v, 4) for k, v in r.items()}, round(time.time() - t0), "s", flush=True)
        if best is None or r["logloss"] < best[0]["logloss"]: best = (r, alpha, clf, p)
    r, alpha, clf, p = best
    nb = ComplementNB(alpha=0.1).fit(Xtr, tr.label.values); pnb = nb.predict_proba(Xte)[:, 1]
    te[f"p_{name}"] = p
    by = te.groupby("len_bin", observed=True).apply(lambda g: pd.Series({
        "n": len(g), "auc": roc_auc_score(g.label, g[f"p_{name}"]), "acc": accuracy_score(g.label, g[f"p_{name}"] > .5),
        "acc_hk": accuracy_score(g.label[g.label == 1], g[f"p_{name}"][g.label == 1] > .5),
        "acc_tw": accuracy_score(g.label[g.label == 0], g[f"p_{name}"][g.label == 0] > .5),
        "frac_confident": ((g[f"p_{name}"] >= .8) | (g[f"p_{name}"] <= .2)).mean(),
        "acc_confident": accuracy_score(g.label[(g[f"p_{name}"] >= .8) | (g[f"p_{name}"] <= .2)],
                                        g[f"p_{name}"][(g[f"p_{name}"] >= .8) | (g[f"p_{name}"] <= .2)] > .5)}))
    print(by.round(4).to_string(), flush=True)
    report[name] = {"alpha": alpha, **r, "nb_auc": roc_auc_score(te.label, pnb), "nb_acc": accuracy_score(te.label, pnb > .5),
                    "by_length": by.round(4).reset_index().to_dict("records")}
    np.savez_compressed(f"{OUT}/hkness_{name}.npz", coef=clf.coef_[0].astype(np.float32), intercept=np.float64(clf.intercept_[0]),
                        mask_terms=np.array(HK_PLACES + TW_PLACES if mask else [], dtype="U"))
te.drop(columns=["text"]).to_parquet(f"{DATA}/test_preds.parquet")
te.to_parquet(f"{DATA}/test_with_text.parquet")
json.dump(report, open(f"{OUT}/eval.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "by_length"} for k, v in report.items()}, indent=1))
