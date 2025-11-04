import os
import io
import csv
import time
import json
import hashlib
import requests
import pandas as pd
import streamlit as st
from diskcache import Cache

# ============================================================
# FIXED SETTINGS (matches Colab defaults)
# ============================================================
NUM_RESULTS = 10
OUTPUT_CSV = "serp_top10_wide.csv"
RETRIES = 5
BASE_SLEEP = 1.0
PER_REQUEST_DELAY = 0.0

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(page_title="Top 10 SERP Results by Keyword (Google)", layout="wide")
st.title("🔍 Top 10 SERP Results by Keyword (Google)")
st.caption("Upload a CSV of keywords, fetch top 10 URLs (and optional titles) via Google Programmable Search API, with caching & resume.")

# ============================================================
# CREDENTIALS FROM SECRETS (no sidebar input)
# ============================================================
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))
CX = st.secrets.get("GOOGLE_CX", os.getenv("GOOGLE_CX", ""))

# ============================================================
# SIDEBAR: Location, Language, Titles toggle
# ============================================================
with st.sidebar:
    st.header("Settings")
    GL = st.text_input("GL (geo bias)", value=os.getenv("GL", "uk"))   # e.g. 'uk'
    HL = st.text_input("HL (UI language)", value=os.getenv("HL", "en"))  # e.g. 'en'
    INCLUDE_TITLES = st.toggle("Also fetch page titles", value=True, help="Adds title_1 … title_10 columns")
    st.caption("API key & CX are read from secrets. Results per query fixed at 10 (Colab parity).")

# ============================================================
# CACHE & CHECKPOINT SETUP
# ============================================================
CACHE_DIR = os.getenv("SERP_CACHE_DIR", ".serp_cache")
CHECKPOINT_DIR = os.getenv("SERP_CHECKPOINT_DIR", ".serp_checkpoints")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
cache = Cache(CACHE_DIR)

BASE = "https://customsearch.googleapis.com/customsearch/v1"
session = requests.Session()
session.headers.update({"Accept-Encoding": "gzip", "User-Agent": "streamlit-pse-wide"})

# ============================================================
# HELPERS
# ============================================================
def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def _key_hash(api_key: str) -> str:
    return _sha256(api_key or "NO_KEY")

def _run_fingerprint(file_bytes: bytes, cx: str, gl: str, hl: str, num_results: int, include_titles: bool) -> str:
    m = hashlib.sha256()
    m.update(file_bytes or b"")
    m.update((cx or "").encode())
    m.update((gl or "").encode())
    m.update((hl or "").encode())
    m.update(str(num_results).encode())
    m.update(b"titles_on" if include_titles else b"titles_off")
    return m.hexdigest()

def _checkpoint_path(run_id: str) -> str:
    return os.path.join(CHECKPOINT_DIR, f"{run_id}.csv")

def _make_query_key(q: str, api_key_hash: str, cx: str, gl: str, hl: str, num_results: int, include_titles: bool) -> str:
    return json.dumps(
        {"q": q, "k": api_key_hash, "cx": cx, "gl": gl, "hl": hl, "n": num_results, "titles": include_titles},
        sort_keys=True,
        ensure_ascii=False,
    )

# ============================================================
# CORE FUNCTIONS
# ============================================================
def load_keywords_from_csv(file_bytes: bytes) -> list[str]:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), dtype=str, keep_default_na=False)
        cols_lower = [c.lower() for c in df.columns]
        col = df.columns[cols_lower.index("keyword")] if "keyword" in cols_lower else df.columns[0]
        kws = df[col].astype(str).str.strip().tolist()
    except Exception:
        kws = []
        for row in csv.reader(io.StringIO(file_bytes.decode("utf-8"))):
            if row and str(row[0]).strip():
                kws.append(str(row[0]).strip())

    seen, out = set(), []
    for k in kws:
        if k and k not in seen:
            seen.add(k); out.append(k)
    return out

def params(q, api_key, cx, gl, hl, num_results, include_titles: bool):
    fields = "items(link,title)" if include_titles else "items(link)"
    return {"key": api_key, "cx": cx, "q": q, "num": num_results, "gl": gl, "hl": hl, "fields": fields}

def _normalize_result(raw, num_results: int, include_titles: bool):
    links = [""] * num_results
    titles = [""] * num_results
    if isinstance(raw, dict):
        items = (raw.get("items") or [])[:num_results]
        for i, it in enumerate(items):
            links[i] = it.get("link", "")
            if include_titles:
                titles[i] = it.get("title", "")
    return {"links": links, "titles": titles}

def fetch_top10_items(q, api_key, cx, gl, hl, num_results, include_titles, retries=RETRIES, base_sleep=BASE_SLEEP):
    fallback = {"links": ["ERROR"] + [""] * (num_results - 1), "titles": [""] * num_results}
    for attempt in range(retries):
        try:
            r = session.get(BASE, params=params(q, api_key, cx, gl, hl, num_results, include_titles), timeout=30)
            if r.status_code == 429:
                time.sleep(base_sleep * (2**attempt) + 0.5)
                continue
            r.raise_for_status()
            data = r.json() or {}
            if isinstance(data, dict) and data.get("error"):
                return fallback
            return _normalize_result(data, num_results, include_titles)
        except requests.exceptions.RequestException:
            if attempt == retries - 1:
                return fallback
            time.sleep(base_sleep * (2**attempt) + 0.5)
    return fallback

def fetch_top10_items_persistent(
    q, api_key, cx, gl, hl,
    num_results=NUM_RESULTS, include_titles=True,
    ttl_seconds=60*60*24*7, retries=RETRIES, base_sleep=BASE_SLEEP,
):
    key = _make_query_key(q, _key_hash(api_key), cx, gl, hl, num_results, include_titles)
    cached = cache.get(key)
    if cached is not None:
        return cached
    result = fetch_top10_items(q, api_key, cx, gl, hl, num_results, include_titles, retries, base_sleep)
    if result["links"] and result["links"][0] != "ERROR":
        cache.set(key, result, expire=ttl_seconds)
    return result

def save_checkpoint(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, path)

# ============================================================
# UPLOAD UI
# ============================================================
uploaded_file = st.file_uploader("Upload CSV of keywords", type=["csv"])
if not uploaded_file:
    st.stop()

file_bytes = uploaded_file.read()
keywords = load_keywords_from_csv(file_bytes)
st.success(f"Loaded {len(keywords)} keywords.")
st.dataframe(pd.DataFrame({"keyword": keywords[:50]}), use_container_width=True, height=200)
if len(keywords) > 50:
    st.caption(f"Showing first 50 of {len(keywords)} keywords.")

run = st.button("Run search")
if not run:
    st.stop()

# ============================================================
# SAFETY CHECKS (secrets only)
# ============================================================
if not API_KEY or API_KEY.startswith("YOUR_"):
    st.error("Please provide GOOGLE_API_KEY in .streamlit/secrets.toml")
    st.stop()
if not CX or CX.startswith("YOUR_"):
    st.error("Please provide GOOGLE_CX in .streamlit/secrets.toml")
    st.stop()

# ============================================================
# EXECUTION LOGIC
# ============================================================
run_id = _run_fingerprint(file_bytes, CX, GL, HL, NUM_RESULTS, INCLUDE_TITLES)
ckpt_path = _checkpoint_path(run_id)

url_cols = [f"rank_{i}" for i in range(1, 11)]
title_cols = [f"title_{i}" for i in range(1, 11)] if INCLUDE_TITLES else []
cols = ["keyword"] + url_cols + title_cols

if os.path.exists(ckpt_path):
    df = pd.read_csv(ckpt_path, dtype=str, keep_default_na=False)
    if set(df.columns) != set(cols):
        df = pd.DataFrame(columns=cols)
    st.info(f"Resuming previous run ({len(df)} rows) for this file/settings.")
else:
    df = pd.DataFrame(columns=cols)

done = set(df["keyword"].astype(str).tolist()) if not df.empty else set()
rows = df.to_dict(orient="records")
error_count = 0 if df.empty else df.get("rank_1", pd.Series([], dtype=str)).eq("ERROR").sum()
total = len(keywords)

progress = st.progress(0)
status = st.empty()
table_placeholder = st.empty()

for i, kw in enumerate(keywords, start=1):
    if kw in done:
        progress.progress(i / total)
        continue

    status.write(f"Fetching top 10 results for: **{kw}** ({i}/{total})")
    result = fetch_top10_items_persistent(
        kw, API_KEY, CX, GL, HL,
        num_results=NUM_RESULTS, include_titles=INCLUDE_TITLES,
        retries=RETRIES, base_sleep=BASE_SLEEP,
    )

    links = result["links"]
    titles = result["titles"] if INCLUDE_TITLES else [""] * NUM_RESULTS

    if links and links[0] == "ERROR":
        error_count += 1

    row = {"keyword": kw}
    for r in range(NUM_RESULTS):
        row[f"rank_{r+1}"] = links[r]
    if INCLUDE_TITLES:
        for r in range(NUM_RESULTS):
            row[f"title_{r+1}"] = titles[r]
    rows.append(row)

    if (i % 10 == 0) or (i == total):
        df = pd.DataFrame(rows, columns=cols)
        table_placeholder.dataframe(df.tail(25), use_container_width=True)
        save_checkpoint(df, ckpt_path)

    progress.progress(i / total)
    if PER_REQUEST_DELAY > 0:
        time.sleep(PER_REQUEST_DELAY)

df = pd.DataFrame(rows, columns=cols)
save_checkpoint(df, ckpt_path)
st.success(f"✅ Completed {len(df)} keywords. Errors: {error_count}")

# ============================================================
# DOWNLOAD OUTPUT
# ============================================================
csv_bytes = df.to_csv(index=False, encoding="utf-8").encode("utf-8")
if st.download_button("⬇️ Download CSV", csv_bytes, OUTPUT_CSV, "text/csv"):
    try:
        os.remove(ckpt_path)
    except FileNotFoundError:
        pass

st.dataframe(df, use_container_width=True)


