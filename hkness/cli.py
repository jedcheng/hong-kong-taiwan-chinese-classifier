"""hkness-score INPUT [--col text] [--model style|full] [--out OUT]
INPUT: .parquet / .csv / .jsonl / .txt (one text per line). Adds p_hk and hk_tw columns."""
import argparse
import pandas as pd
from . import Scorer


def main():
    ap = argparse.ArgumentParser(description="Score Traditional Chinese text for Hong Kong vs Taiwan style.")
    ap.add_argument("input")
    ap.add_argument("--col", default="text")
    ap.add_argument("--model", default="style", choices=["style", "full"])
    ap.add_argument("--out")
    ap.add_argument("--batch", type=int, default=50000)
    a = ap.parse_args()
    ext = a.input.rsplit(".", 1)[-1]
    if ext == "parquet": df = pd.read_parquet(a.input)
    elif ext == "csv": df = pd.read_csv(a.input)
    elif ext == "jsonl": df = pd.read_json(a.input, lines=True)
    else: df = pd.DataFrame({a.col: [l.rstrip("\n") for l in open(a.input, encoding="utf-8") if l.strip()]})
    s = Scorer(a.model)
    texts = df[a.col].fillna("").astype(str).tolist()
    p = []
    for i in range(0, len(texts), a.batch):
        p.extend(s.score(texts[i:i + a.batch]))
    df["p_hk"] = p
    df["hk_tw"] = ["HK" if x >= .8 else "TW" if x <= .2 else "uncertain" for x in p]
    out = a.out or a.input.rsplit(".", 1)[0] + f"_hkness_{a.model}.csv"
    df.to_parquet(out, index=False) if out.endswith(".parquet") else df.to_csv(out, index=False)
    print(f"wrote {out}  n={len(df)}  mean p_hk={df.p_hk.mean():.3f}  {df.hk_tw.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
