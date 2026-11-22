# Hong Kong vs Taiwan Chinese Classifier (`hkness`)

Scores how **Hong Kong-like** or **Taiwan-like** a piece of Traditional Chinese text is.

Hong Kong and Taiwan both write Traditional Chinese, but their vocabulary, register and character
variants differ: 軟件/軟體, 網上/線上, 質素/品質, 的士/計程車, 現時/目前, 裏/裡, 着/著.
`hkness` turns those differences into one number, `p_hk`, the probability that a text was written
in Hong Kong style rather than Taiwan style.

- **Fast and light.** A character 1–3-gram logistic regression with no tokenizer, no GPU and no
  jieba. It scores about 5,500 docs/s (1.1M chars/s) per CPU core. Each model is a 16 MB `.npz` file.
- **Interpretable.** `explain()` lists the n-grams that pushed a text toward HK or TW.
- **Not fooled by topic.** The default `style` model masks HK/TW place and polity names (香港, 銅鑼灣,
  台北, 立法院…), so a Taiwanese article about Hong Kong is still scored as Taiwanese.
- **Trained on 905k web snippets** from `.hk` and `.tw` sites in Common Crawl 2025–2026, and
  **evaluated on websites held out from training**.

Model weights: [huggingface.co/jed351/hong-kong-taiwan-chinese-classifier](https://huggingface.co/jed351/hong-kong-taiwan-chinese-classifier)
Code: [github.com/jedcheng/hong-kong-taiwan-chinese-classifier](https://github.com/jedcheng/hong-kong-taiwan-chinese-classifier)

## Install

```bash
pip install git+https://github.com/jedcheng/hong-kong-taiwan-chinese-classifier
```
Dependencies: numpy, scikit-learn, pandas, huggingface_hub. The model downloads from the Hugging Face
Hub on first use.

## Usage

```python
from hkness import Scorer

s = Scorer()          # "style" model (place names masked); Scorer("full") = unmasked
s.score([
    "政府表示，現時有關措施有助減低市民的開支，詳情可瀏覽網上平台或致電查詢熱線。",
    "政府表示，目前相關措施有助降低民眾的開銷，詳細資訊可上網路平台或撥打專線洽詢喔。",
])
# array([1.0000, 0.0009])      P(HK): 0 = Taiwan-like, 1 = Hong Kong-like

labels, p = s.label(texts)   # "HK" (p >= 0.8) / "TW" (p <= 0.2) / "uncertain"

s.explain("目前相關措施有助降低民眾的開銷喔", k=3)
# {'logit': -3.77, 'toward_TW': [('民眾', -2.21), ('喔', -1.59), ('目前', -0.91), ...],
#  'toward_HK': [('有助', 1.49), ...]}
```

Command line (parquet / csv / jsonl / txt with one text per line):
```bash
hkness-score my_texts.jsonl --col text            # writes my_texts_hkness_style.csv with p_hk, hk_tw
```

`p_hk` is calibrated on balanced classes, so it assumes HK and TW are equally likely a priori.

## Evaluation

The test set comes from **websites that were never used in training** (10% of domains held out):
75,150 snippets, balanced HK/TW. Results for the default `style` model:

| text length (chars) | AUC | accuracy | share scored confidently (p ≤ .2 or ≥ .8) | accuracy when confident |
|---|---|---|---|---|
| ≤ 50 | 0.889 | 80.4% | 53% | 92.5% |
| 50–100 | 0.925 | 84.5% | 62% | 94.4% |
| 100–200 | 0.947 | 87.4% | 70% | 95.2% |
| 200–500 | 0.965 | 89.9% | 78% | 96.2% |
| 500+ | 0.974 | 91.4% | 83% | 96.7% |
| **all** | **0.940** | **86.2%** | | |

- The `full` model (place names not masked) gets AUC 0.943, so place names add very little.
- A Complement Naive Bayes baseline on the same features gets AUC 0.904.
- Labels come from the site's domain (.hk / .tw), so some of the remaining error is label noise
  (HK sites republishing Taiwanese text, and the reverse).
- Swapping Taiwanese terms for their HK equivalents (from a list of 57 known pairs) in 7.8k held-out
  Taiwanese snippets raised `p_hk` in 99.9% of them.

## Training data

- **Source:** the 2025–2026 dumps (22 crawls, CC-MAIN-2025-05 … CC-MAIN-2026-43) of a Traditional
  Chinese Common Crawl corpus built with [c4-dataset-script](https://github.com/jedcheng/c4-dataset-script),
  MinHash-deduplicated across dumps. See [jed351/Traditional-Chinese-Web-Data](https://huggingface.co/datasets/jed351/Traditional-Chinese-Web-Data).
- **Region label:** the hostname's top-level domain. `.hk` includes .com.hk, .org.hk, .edu.hk…; the
  same goes for `.tw`. That gives 2.29M HK docs (19.7k domains) and 15.3M TW docs (81k domains).
- **Written Chinese only:** docs that [CantoneseDetect](https://github.com/CanCLID/cantonesedetect)
  labels `swc` or `neutral`. Cantonese docs (~11% of .hk) are excluded, so the model learns HK
  *written* Chinese. Cantonese input still scores as HK (e.g. 我哋今日去咗食嘢 → 0.94).
- **Cleaning:** at most about 300 docs per domain, so big sites don't dominate, and online-casino SEO
  pages removed (common on .tw).
- **Snippets:** one random window per doc, with length log-uniform between 30 and 1,500 characters,
  so the model works on both short and long inputs.
- **Model:** binary char 1–3-gram hashing (2^22 features, L2-normalized, digits → 0), logistic
  regression trained with SGD (log loss, alpha = 3e-7, chosen on test log-loss).

## Regional lexicon

`lexicon/` holds the word-level analysis behind the model. The same .hk/.tw corpus was segmented with
jieba (`dict.txt.big` plus a user dictionary with both regions' variant spellings), and each word
was scored by document-frequency log-odds with an informative Dirichlet prior (Monroe et al., 2008).

| file | content |
|---|---|
| `lexicon_written.csv.gz` | 108k words with HK/TW document frequency, rate per 10k docs, `hk_share` = P(HK \| word) under equal priors, log2 ratio, z-score, domain spread |
| `top_hk_written.csv`, `top_tw_written.csv` | top 500 words per side (≥ 4× skew, spread over ≥ 10 domains) |
| `pairs_validation.csv` | 57 known HK/TW term pairs: usage of each form in each region |
| `char_variants.csv` | HK vs TW standard character forms (裏/裡, 綫/線, 着/著, 衞/衛…) |

Some of the most distinctive words:
- **HK:** 現時, 作出, 令, 致力, 早前, 閣下, 毋須, 視乎, 會否, 達致, 港幣, 按揭, 樓宇, 電郵, 戶口, 義工, 私隱, 質素.
- **TW:** 網路, 軟體, 帳號, 簡訊, 機車, 醫師, 民眾, 業者, 計畫, 依法, 宣導, 建置, 喔, 囉, 蠻.

The split is lopsided. HK web text often uses Taiwanese forms (手機, 螢幕, 品質 and 預設 are the
majority even on .hk sites), but Taiwanese web text almost never uses HK forms.

## Limitations

- **Web-register style.** HK officialese (現時, 致力, 閣下) is a strong HK cue, so casual HK
  writing that avoids it gets a less extreme score.
- **Short texts.** Under 50 characters the model is weak; use the `uncertain` band.
- **Traditional characters only.** Simplified characters lean TW, because some .tw pages are
  machine-converted.
- **Mainland Chinese** written in Traditional characters is out of scope; it was never a class.
- **Domain labels are noisy**, and web text over-represents commercial and SEO content.

## Reproducing

`training/` contains the full pipeline: `extract.py` → `tokenize_count.py` → `analyze.py` for the
lexicon, and `build_data.py` → `train.py` for the classifier. Set `HKTW_SRC` (a folder of
`<dump>/*.parquet` files with `url,timestamp,text`), `HKTW_DATA` and `HKTW_WS`. The jieba run also
needs `dict/dict.txt.big` from the jieba repo and `make_userdict.py`. The original run took about
45 minutes for the lexicon and 7 minutes for the classifier on a 36-core node.

## License

Apache-2.0 for code and model weights. The training text comes from Common Crawl and is not
redistributed here.

## Citation

```bibtex
@misc{cheng2026hkness,
  title  = {Hong Kong vs Taiwan Chinese Classifier},
  author = {Cheng, Jed},
  year   = {2026},
  url    = {https://github.com/jedcheng/hong-kong-taiwan-chinese-classifier}
}
```
