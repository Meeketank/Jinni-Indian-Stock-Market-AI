# --- SYMBOL RESOLUTION (robust) ---
import json

YAHOO_SEARCH_CACHE = os.path.join(CACHE_DIR, "yahoo_search_cache.pkl")
# load cache
try:
    with open(YAHOO_SEARCH_CACHE, "rb") as f:
        _yahoo_cache = pickle.load(f)
except Exception:
    _yahoo_cache = {}

def save_yahoo_cache():
    try:
        with open(YAHOO_SEARCH_CACHE, "wb") as f:
            pickle.dump(_yahoo_cache, f)
    except Exception:
        pass

def yahoo_search_symbol(query: str) -> List[str]:
    """
    Query Yahoo Finance search API for best ticker suggestions.
    Returns list of candidate tickers (ordered), e.g. ['IDBI.NS', 'IDBI.BO']
    Uses a small local cache to prevent repeated network calls.
    """
    q = (query or "").strip().upper()
    if not q:
        return []
    if q in _yahoo_cache:
        return _yahoo_cache[q]
    # safe network call with small backoff
    try:
        throttle_fetch()
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        r = requests.get(url, params={"q": q, "quotesCount": 10, "newsCount": 0}, timeout=6)
        candidates = []
        if r.status_code == 200:
            j = r.json()
            for itm in j.get("quotes", []) + j.get("news", []):
                sym = itm.get("symbol")
                exch = itm.get("exchange")
                # prefer NSE (.NS) then BSE (.BO)
                if not sym:
                    continue
                # direct symbol e.g. "IDBI.NS" is fine
                candidates.append(sym.upper())
            # dedupe keep order
            seen = set(); out=[]
            for s in candidates:
                if s not in seen:
                    out.append(s); seen.add(s)
            candidates = out
        else:
            candidates = []
    except Exception:
        candidates = []
    # fallback to simple candidates if none found
    if not candidates:
        simple = [q, q + ".NS", q + ".BO", q + ".NSE", q + ".BSE"]
        candidates = [s for s in simple]
    # cache and return
    _yahoo_cache[q] = candidates
    save_yahoo_cache()
    return candidates

def resolve_and_fetch(symbol_input: str, period="2y") -> Tuple[pd.DataFrame, str]:
    """
    Try several resolution steps:
     - If user already included suffix, try that first
     - Try yahoo_search_symbol suggestions (prefers official tickers)
     - Try manual variants (.NS, .BO, uppercase)
    Returns (hist_df or None, resolved_symbol or None)
    """
    s = (symbol_input or "").strip()
    if not s:
        return None, None
    s_up = s.upper()
    tried = []
    # 1) Direct attempt if looks like a ticker with suffix
    candidates = []
    if "." in s_up:
        candidates.append(s_up)
    # 2) Add common suffixes and raw
    for v in [s_up, s_up + ".NS", s_up + ".BO", s_up + ".NSE", s_up + ".BSE"]:
        if v not in candidates:
            candidates.append(v)
    # 3) Yahoo search suggestions (try these before exhausting more calls)
    try:
        yahoo_cands = yahoo_search_symbol(s_up)
        # prepend yank suggestions so they are tried first
        for yc in yahoo_cands:
            if yc not in candidates:
                candidates.insert(0, yc)
    except Exception:
        pass

    # try candidates until success, record last error for debugging
    last_err = None
    for c in candidates:
        if c in tried: 
            continue
        tried.append(c)
        try:
            # use cached hist if present
            hist = load_cache(cache_path(c, "hist"), FETCH_TTL)
            if hist is not None:
                return hist, c
            # else fetch (throttled)
            throttle_fetch()
            tk = yf.Ticker(c)
            df = tk.history(period=period, interval="1d", auto_adjust=True)
            if df is not None and not df.empty:
                # cache and return
                save_cache(df, cache_path(c, "hist"))
                return df, c
            else:
                last_err = f"No data for {c}"
        except Exception as e:
            last_err = str(e)
            # slight pause if we hit transient limits
            time.sleep(0.2)
            continue

    # If we get here, nothing worked
    # save last error for diagnostics
    with open(os.path.join(CACHE_DIR, "last_symbol_resolution.log"), "a") as f:
        f.write(f"\n[{datetime.now().isoformat()}] Failed resolving '{symbol_input}'. Tried: {candidates}. Last err: {last_err}\n")
    return None, None

# -------------------- Replace previous "resolve candidates" and fetch logic with this --------------------
# Example usage (drop in where you previously resolved/fetched 'hist'):
hist, resolved = resolve_and_fetch(user_input, period="2y")
if hist is None:
    st.error(f"Could not fetch data for that symbol. Tried many variants and Yahoo suggestions. Try adding .NS or .BO suffix or check spelling. (Tried: {', '.join(yahoo_search_symbol(user_input))})")
    st.stop()

# add to BG universe
BG.add_symbol(resolved)
# continue with indicator calc, plotting etc...
hist = add_indicators(hist)
