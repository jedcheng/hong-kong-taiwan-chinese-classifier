"""hkness: how Hong Kong-like vs Taiwan-like a piece of Traditional Chinese text is.

    from hkness import Scorer
    s = Scorer()                    # "style" model (HK/TW place & polity names masked)
    s.score(["現時有關措施有助減低市民的開支"])   # -> array([0.99998]) = P(HK), equal priors
    s.label(texts)                  # ("HK" | "TW" | "uncertain", p)
    s.explain(text, k=10)           # char n-grams pushing toward HK / TW
"""
import math, os, re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

__version__ = "0.1.0"
HF_REPO = "jed351/hong-kong-taiwan-chinese-classifier"
N_FEATURES = 2 ** 22
MASK_TOKEN = "□"
DIGIT_RE = re.compile(r"[0-9０-９]+")
WS_RE = re.compile(r"\s+")


def make_vectorizer():
    return HashingVectorizer(analyzer="char", ngram_range=(1, 3), n_features=N_FEATURES,
                             alternate_sign=False, binary=True, norm="l2", lowercase=False)


def _resolve(model, path):
    if path:
        return path
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"hkness_{model}.npz")
    if os.path.exists(local):
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, f"hkness_{model}.npz")


class Scorer:
    """model: "style" (default; place/polity names masked) or "full" (unmasked).
    path: a local .npz; otherwise the file is fetched from the Hugging Face Hub."""

    def __init__(self, model="style", path=None):
        z = np.load(_resolve(model, path), allow_pickle=False)
        self.coef = z["coef"].astype(np.float32)
        self.intercept = float(z["intercept"])
        terms = [str(t) for t in z["mask_terms"]]
        self.mask_re = re.compile("|".join(sorted(map(re.escape, terms), key=len, reverse=True))) if terms else None
        self.vec = make_vectorizer()

    def normalize(self, text):
        t = WS_RE.sub(" ", DIGIT_RE.sub("0", text.lower()))
        return self.mask_re.sub(MASK_TOKEN, t) if self.mask_re else t

    def logit(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        X = self.vec.transform([self.normalize(t) for t in texts])
        return X @ self.coef + self.intercept

    def score(self, texts):
        """P(HK) assuming equal priors. 0 = Taiwan-like, 1 = Hong Kong-like."""
        return 1.0 / (1.0 + np.exp(-self.logit(texts)))

    def label(self, texts, lo=0.2, hi=0.8):
        p = self.score(texts)
        return np.where(p >= hi, "HK", np.where(p <= lo, "TW", "uncertain")), p

    def explain(self, text, k=10):
        """Distinct char n-grams with the largest contributions (weight x feature value).
        All contributions + intercept sum to logit(text) (up to rare hash collisions)."""
        from sklearn.utils import murmurhash3_32
        t = WS_RE.sub(" ", self.normalize(text))
        grams = {t[i:i + n] for n in (1, 2, 3) for i in range(len(t) - n + 1)}
        val = 1 / math.sqrt(len(grams)) if grams else 0.0
        out = sorted(((g, float(self.coef[abs(murmurhash3_32(g, seed=0)) % N_FEATURES] * val)) for g in grams),
                     key=lambda x: x[1])
        return {"logit": float(self.logit(text)[0]), "toward_TW": out[:k], "toward_HK": out[::-1][:k]}
