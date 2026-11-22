from urllib.parse import urlsplit
SLD = {"com","org","edu","gov","net","idv","game","club","ebiz"}
def host_info(url):
    """-> (host, domain, region) ; region in {'hk','tw',None} by hostname TLD."""
    h = (urlsplit(url).hostname or "").lower().rstrip(".")
    if h.startswith("www."): h = h[4:]
    p = h.split(".")
    tld = p[-1]
    region = tld if tld in ("hk", "tw") else None
    n = 3 if len(p) >= 3 and p[-2] in SLD and region else 2
    return h, ".".join(p[-n:]), region

import os
# Configure with env vars. SRC = folder of <dump>/*.parquet with url,timestamp,text columns
# (e.g. a local copy of jed351/Traditional-Chinese-Web-Data, 2025-2026 dumps).
SRC = os.environ.get("HKTW_SRC", "./cc_parquet")
DATA = os.environ.get("HKTW_DATA", "./data")   # outputs (tens of GB)
WS = os.environ.get("HKTW_WS", ".")              # holds dict/ (jieba dict.txt.big + user_dict.txt) and out/
CAP = 300  # max expected docs per registered domain in the analysis sample

# HK-standard form <-> TW-standard form character variants (used to augment the jieba dict
# so that neither spelling gets split just because the dictionary only lists the other one)
VARIANTS = [("裏","裡"),("綫","線"),("着","著"),("衞","衛"),("麪","麵"),("峯","峰"),("羣","群"),
            ("啓","啟"),("爲","為"),("僞","偽"),("衆","眾"),("温","溫"),("説","說"),("污","汙"),
            ("牀","床"),("綉","繡"),("鷄","雞"),("册","冊"),("濕","溼"),("擡","抬"),("箇","個"),
            ("嫺","嫻"),("粧","妝"),("脣","唇"),("蔥","葱"),("鈎","鉤"),("衹","只"),("黏","粘"),
            ("鏽","銹"),("戶","户"),("内","內"),("吿","告"),("强","強"),
            ("够","夠"),("敎","教"),("麽","麼"),("巿","市")]

def keep_prob(n_domain_docs):
    return min(1.0, CAP / n_domain_docs)

def stable_unit(s):
    import hashlib
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:8], "big") / 2**64

def init_jieba():
    import jieba
    jieba.setLogLevel(60)
    jieba.set_dictionary(f"{WS}/dict/dict.txt.big")
    jieba.load_userdict(f"{WS}/dict/user_dict.txt")
    jieba.initialize()
    return jieba
