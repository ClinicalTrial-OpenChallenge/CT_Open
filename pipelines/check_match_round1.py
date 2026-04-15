import pickle
from tqdm import tqdm
import random
import sys
from generation import generate
# sys.path.append('/data/rmap/')
# auto reload
#reload_ext autoreload
#autoreload 2
from collections.abc import Iterable
from typing import Optional
import re
from datetime import date
from typing import Any, Iterable, List
import tiktoken

_PARTIAL_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")

# Common dash/hyphen characters (copy-pasteable)
COMMON_DASHES = [
    "-",   # U+002D HYPHEN-MINUS (ASCII)
    "‐",   # U+2010 HYPHEN
    "-",   # U+2011 NON-BREAKING HYPHEN
    "–",   # U+2013 EN DASH
    "—",   # U+2014 EM DASH
    "―",   # U+2015 HORIZONTAL BAR
    "−",   # U+2212 MINUS SIGN (math)
    "‒",   # U+2012 FIGURE DASH
]
def normalize_common_dashes(text: str) -> str:
    dash_map = dict.fromkeys(map(ord, "".join(COMMON_DASHES)), ord("-"))
    return text.translate(dash_map)

def get_all_urls_set(
    *,
    dbname: str = "web_database",
    user: str = "postgres",
    password: str = None,
    host: str = "/tmp",
    port: int = None,
    fetch_size: int = 10000,
) -> set[str]:
    """
    Stream all URLs from scrapes into a Python set efficiently.
    - Uses a server-side (named) cursor to avoid loading all rows at once.
    - No ORDER BY to let Postgres choose the fastest path (often index-only).
    """
    import psycopg2

    urls: set[str] = set()
    connect_kwargs = dict(dbname=dbname, user=user, host=host)
    if password:
        connect_kwargs["password"] = password
    if port:
        connect_kwargs["port"] = port
    conn = psycopg2.connect(**connect_kwargs)
    try:
        # Named cursors require a transaction; make it read-only for safety.
        conn.set_session(readonly=True, autocommit=False)

        with conn.cursor(name="url_stream") as cur:
            cur.itersize = fetch_size  # batch size per round trip
            cur.execute("SELECT url FROM scrapes WHERE url IS NOT NULL;")
            for (u,) in cur:
                urls.add(u)
    finally:
        conn.close()
    return urls

def get_url_metadata_dict(
    urls: Iterable[str],
    *,
    dbname: str = "web_database",
    user: str = "postgres",
    password: str = None,
    host: str = "/tmp",
    port: int = None,
    batch_size: int = 10_000,
    fetch_size: int = 10_000,
) -> dict[str, tuple[Optional[str], Optional[object]]]:
    """
    For a given iterable of URLs, return a dict: {url: (longfilatura, pub_date)}.

    - Any URL not present in the DB is included with value (None, None).
    - Processes input URLs in batches of `batch_size` to keep SQL parameters bounded.
    - Uses a server-side (named) cursor and `itersize` to stream results efficiently.

    Table expected: scrapes(url PRIMARY KEY/INDEXED, longfilatura TEXT, pub_date DATE/TIMESTAMP)

    Returns
    -------
    dict[str, tuple[Optional[str], Optional[date|datetime]]]
    """
    import psycopg2

    # Materialize and normalize the input once (dedupe but preserve the requirement
    # to include missing URLs in the result).
    url_list = list(dict.fromkeys(u for u in urls if u))  # dedupe while preserving order

    # Pre-fill with (None, None) for URLs not found later
    result: dict[str, tuple[Optional[str], Optional[object]]] = {u: (None, None) for u in url_list}

    if not url_list:
        return result

    connect_kwargs = dict(dbname=dbname, user=user, host=host)
    if password:
        connect_kwargs["password"] = password
    if port:
        connect_kwargs["port"] = port
    conn = psycopg2.connect(**connect_kwargs)    
    try:
        # Read-only transaction for safety; autocommit False is required for named cursors
        conn.set_session(readonly=True, autocommit=False)

        # Process in batches of up to `batch_size` URLs
        for i in range(0, len(url_list), batch_size):
            batch = url_list[i : i + batch_size]

            # Server-side cursor name must be unique per use
            cursor_name = f"meta_stream_{i//batch_size}"

            # Use url = ANY(%s) so we bind a single list parameter
            # date exist as pub_date in the db
            # TODO: get paper or pdf if exist instead of longfilatura
            with conn.cursor(name=cursor_name) as cur:
                cur.itersize = fetch_size
                cur.execute(
                    """
                    SELECT url, longfilatura, paper,pdf
                    FROM scrapes
                    WHERE url = ANY(%s)
                    """,
                    (batch,),
                )
                for row in cur:
                    u, longfilatura, paper, pdf = row
                    if paper is not None:
                        result[u] = paper
                    elif pdf is not None:
                        result[u] = pdf
                    else:
                        result[u] = longfilatura
                    
                    if paper is not None and paper != longfilatura:
                        result[u] = longfilatura + paper
                    elif pdf is not None and pdf != longfilatura:
                        result[u] = longfilatura + pdf

        # Commit is not strictly necessary for a read-only transaction, but close cleanly
        conn.rollback()  # ensure we leave no open transaction
    finally:
        conn.close()

    return result


def strip_dates_from_trial(data):
    """
    Return a deep-copied dictionary with all date information removed.
    Removes:
      - Any dict entries whose keys look date-related (e.g., updatedAt, last_verified, primaryCompletionDate)
      - Entire subtrees for certain known date-only structures (e.g., 'startDateStruct')
      - Any scalar string values that are pure dates/timestamps (various common formats)
      - Any list items that are pure dates/timestamps
      - Any empty dicts/lists that result from pruning
    """
    if not isinstance(data, dict):
        raise TypeError("Input must be a dictionary.")



    # --- configuration ---
    EXCLUDED_SUBTREE_KEYS = {"startDateStruct"}

    # normalized-substring checks (on lowercase key with non-alphanumerics removed)
    DATE_SUBSTRINGS = (
        "date",
        "datetime",
        "timestamp",
        "updatedat",
        "createdat",
        "lastupdated",
        "lastupdate",
        "firstposted",
        "lastposted",
        "lastverified",
        "completiondate",
        "primarycompletiondate",
    )

    # token-level indicators once we split key into words/digits
    DATE_TOKENS = {
        "date",
        "datetime",
        "timestamp",
        "posted",
        "verified",
        "updated",
        "created",
    }

    # strict date/timestamp value detectors (full-string matches only)
    ISO_DATE_RE = re.compile(
        r"^\d{4}-\d{2}-\d{2}(?:[T\s]\d{2}:\d{2}(?::\d{2})?(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})?)?$"
    )
    ISO_YM_RE = re.compile(r"^\d{4}-\d{2}$")
    SLASH_YMD_RE = re.compile(r"^\d{4}/\d{2}/\d{2}$")
    US_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
    DMY_DASH_RE = re.compile(r"^\d{1,2}-\d{1,2}-\d{4}$")
    TEXT_MONTH_DAY_YEAR_RE = re.compile(
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}$",
        flags=re.IGNORECASE,
    )
    TEXT_MONTH_YEAR_RE = re.compile(
        r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$",
        flags=re.IGNORECASE,
    )

    KEY_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")

    def _normalize_key(k: str) -> str:
        return re.sub(r"[^a-z0-9]", "", k.lower())

    def _key_tokens(k: str) -> List[str]:
        return [t.lower() for t in KEY_TOKEN_RE.findall(k)]

    def looks_like_date_value(val: Any) -> bool:
        if not isinstance(val, str):
            return False
        s = val.strip()
        if not s:
            return False
        if (
            ISO_DATE_RE.match(s)
            or ISO_YM_RE.match(s)
            or SLASH_YMD_RE.match(s)
            or US_DATE_RE.match(s)
            or DMY_DASH_RE.match(s)
            or TEXT_MONTH_DAY_YEAR_RE.match(s)
            or TEXT_MONTH_YEAR_RE.match(s)
        ):
            return True
        return False

    def is_date_key(key: Any, *, date_substrings: Iterable[str] = DATE_SUBSTRINGS, date_tokens: Iterable[str] = DATE_TOKENS) -> bool:
        s = str(key)
        nk = _normalize_key(s)

        # substring checks (e.g., updatedAt -> 'updatedat')
        for sub in date_substrings:
            if sub in nk:
                # avoid false positive for words like "candidate"
                if sub == "date" and "candidate" in nk:
                    if nk.replace("candidate", "", 1).find("date") == -1:
                        continue
                return True

        toks = set(_key_tokens(s))
        if {"date"} & toks or {"datetime"} & toks or {"timestamp"} & toks:
            return True
        if {"first", "posted"}.issubset(toks):
            return True
        if {"last", "posted"}.issubset(toks):
            return True
        if {"last", "verified"}.issubset(toks):
            return True
        if {"primary", "completion"}.issubset(toks):
            return True
        if "updated" in toks and ("at" in toks or "time" in toks or "on" in toks):
            return True
        if toks & set(date_tokens):
            return True
        return False

    def prune(obj: Any):
        if isinstance(obj, dict):
            new = {}
            for k, v in obj.items():
                # drop excluded subtrees outright
                if str(k) in EXCLUDED_SUBTREE_KEYS:
                    continue
                # drop date-like keys
                if is_date_key(k):
                    continue
                # drop pure date/timestamp strings directly stored as the value
                if isinstance(v, str) and looks_like_date_value(v):
                    continue
                pv = prune(v)
                if pv in (None, {}, []):
                    continue
                new[k] = pv
            return new

        if isinstance(obj, list):
            out = []
            for item in obj:
                # drop list items that are pure date/timestamp strings
                if isinstance(item, str) and looks_like_date_value(item):
                    continue
                pv = prune(item)
                if pv in (None, {}, []):
                    continue
                out.append(pv)
            return out

        # scalar (keep as-is unless pruned by parent logic)
        return obj

    return prune(data)

def _parse_partial(d: str):
    """
    Parse 'YYYY', 'YYYY-MM', or 'YYYY-MM-DD' into (year, month, day).
    Missing month/day are returned as None.
    Validates actual dates when day is present.
    """
    m = _PARTIAL_RE.match(d)
    if not m:
        raise ValueError(f"Invalid date format: {d!r}. Use YYYY, YYYY-MM, or YYYY-MM-DD.")
    y, mo, da = m.groups()
    y = int(y)
    if mo is not None:
        mo_i = int(mo)
        if not (1 <= mo_i <= 12):
            raise ValueError(f"Invalid month in {d!r}.")
        if da is not None:
            da_i = int(da)
            # validate actual calendar date
            try:
                date(y, mo_i, da_i)
            except ValueError:
                raise ValueError(f"Invalid day in {d!r}.")
            return y, mo_i, da_i
        return y, mo_i, None
    return y, None, None

def _augment_for_upper_bound(y, m, d):
    """
    Replace missing parts with 'upper bound' sentinels so that the
    partial date sorts AFTER any concrete date in the same period.
    """
    MISSING_MONTH_SENTINEL = 13   # after any real month (1..12)
    MISSING_DAY_SENTINEL   = 32   # after any real day (1..31)
    if m is None:
        return (y, MISSING_MONTH_SENTINEL, MISSING_DAY_SENTINEL)
    if d is None:
        return (y, m, MISSING_DAY_SENTINEL)
    return (y, m, d)

def _require_full_date(d: str):
    """Ensure a full YYYY-MM-DD and return (y,m,day)."""
    y, m, day = _parse_partial(d)
    if m is None or day is None:
        raise ValueError(f"Reference date must be full (YYYY-MM-DD), got {d!r}.")
    return (y, m, day)

def is_before_with_partial(reference_full: str, candidate: str) -> bool:
    """
    Return True if 'candidate' is strictly before 'reference_full' under this rule:
      - Missing parts are treated as the latest possible within their scope,
        so 'YYYY' > any date in that year, and 'YYYY-MM' > any day in that month.
    The 'reference_full' must be a full date 'YYYY-MM-DD'.
    """
    ref = _require_full_date(reference_full)
    y, m, d = _parse_partial(candidate)
    cand_aug = _augment_for_upper_bound(y, m, d)
    return cand_aug < ref

# # --- examples ---
# if __name__ == "__main__":
#     print(is_before_with_partial("2025-02-01", "2024"))       # True
#     print(is_before_with_partial("2025-02-01", "2025"))       # False (treated as after any 2025 date)
#     print(is_before_with_partial("2025-02-01", "2025-01"))    # True
#     print(is_before_with_partial("2025-02-01", "2025-02"))    # False (treated as after any 2025-02 day)
#     print(is_before_with_partial("2025-02-01", "2025-02-01")) # False (equal, not before)
#     print(is_before_with_partial("2025-02-01", "2026"))       # False

def get_oai_encoding(model_name: str = "gpt-4o"):
    """
    Prefer the model-specific encoding; fall back to o200k_base, then cl100k_base
    if your local tiktoken version doesn't recognize gpt-4o.
    """
    try:
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")

def split_by_tokens(text: str, max_tokens: int, enc) -> list[str]:
    """
    Losslessly splits `text` into chunks whose encoded length is <= max_tokens.
    """
    ids = enc.encode(text)
    parts: list[str] = []
    for i in range(0, len(ids), max_tokens):
        chunk_ids = ids[i:i + max_tokens]
        parts.append(enc.decode(chunk_ids))
    return parts

# new function: load the pickle file and get the corresponding nctid you need
def to_lists(pkl_obj: Any):
    """
    return：
      keys_list:   [k1, k2, ...]
      values_list: [v1, v2, ...]
      items_list:  [(k1, v1), (k2, v2), ...]
    """
    if isinstance(pkl_obj, dict):
        keys_list = list(pkl_obj.keys())
        values_list = list(pkl_obj.values())
        items_list = list(pkl_obj.items())
        return keys_list, values_list, items_list

    # if pickle is list/tuple 
    if isinstance(pkl_obj, (list, tuple)):
        return None, None, list(pkl_obj)

    raise TypeError(f"Unsupported pickle type: {type(pkl_obj)}")

failed_pairs = []          # [(nctid, url), ...]
failed_reason = {}  

def _mark_fail(reason: str, nctid: str, url: str):
    failed_pairs.append((nctid, url))
    failed_reason[reason] = failed_reason.get(reason, 0) + 1






def main(args):

    # with open("./data/sep23_earliest_dates_all.pickle","rb") as f:
    #     earliest_dates_all = pickle.load(f)

    with open(args.earliest_dates_path,"rb") as f:
        earliest_dates_all = pickle.load(f)

    # with open("./data/sep25_url2longfilatura.pickle","rb") as f:
    #     url2longfilatura = pickle.load(f)
    
    # with open('./data/sep11_nctid2url_full_36k.pickle','rb') as f:
    #     nctid2url = pickle.load(f)

    with open(args.nctid_to_url_path, 'rb') as f:
        nctid2url = pickle.load(f)

    # with open("./data/sep22_sampled_2000_pairs.pickle","rb") as f:
    #     sample_2000 = pickle.load(f)

    # with open("./data/aug15_36k_clinical_trials.pickle","rb") as f:
    #     all_sampled_urls = pickle.load(f)

    with open(args.all_36k_trials_path,"rb") as f:
        all_36k_trials = pickle.load(f)


    # Gene_therapy_clinical_trials= ['NCT05432882', 'NCT06869278', 'NCT01766739', 'NCT06828042', 'NCT06963008', 'NCT06690359', 'NCT06916767', 'NCT05973487', 'NCT03603405', 'NCT06885697', 'NCT05714904', 'NCT05703971', 'NCT05699811', 'NCT06820424', 'NCT06510374', 'NCT06503497', 'NCT06326008', 'NCT06861452', 'NCT06708845', 'NCT05828212', 'NCT06597656', 'NCT04271644', 'NCT06647329', 'NCT06297226', 'NCT05887167', 'NCT05459571', 'NCT06888648', 'NCT05680922', 'NCT05902962', 'NCT04374136', 'NCT05359211', 'NCT05028933', 'NCT04778579', 'NCT04661020', 'NCT05274451', 'NCT06228924', 'NCT05947487', 'NCT07095686', 'NCT05507827', 'NCT05189925', 'NCT06428188', 'NCT06519344', 'NCT04007029', 'NCT06369974', 'NCT04903080', 'NCT06474416', 'NCT06343311', 'NCT06641154', 'NCT06647979', 'NCT06008925', 'NCT05800977', 'NCT06687837', 'NCT05707273', 'NCT05451849', 'NCT06913608', 'NCT05442515', 'NCT06937567', 'NCT06622694', 'NCT06898970', 'NCT06308159', 'NCT05990621', 'NCT05052957', 'NCT03832855', 'NCT04438083', 'NCT05345171', 'NCT05859074', 'NCT06316856', 'NCT03546361', 'NCT06300476', 'NCT05052528', 'NCT05023889', 'NCT04405778', 'NCT06960213', 'NCT06980597', 'NCT05566223', 'NCT06109181', 'NCT04482933', 'NCT06365671', 'NCT04684563', 'NCT06727721', 'NCT02337985', 'NCT05842707', 'NCT06545201', 'NCT05577312', 'NCT06943937', 'NCT04581473', 'NCT05948033', 'NCT05979792', 'NCT05533697', 'NCT06367673', 'NCT06545955', 'NCT06549296', 'NCT06061549', 'NCT05902598', 'NCT06968195', 'NCT05391490', 'NCT06569472', 'NCT01757223', 'NCT06904729', 'NCT03311503']
    # Skin_therapy_clinical_trials = ['NCT05582434', 'NCT07044141', 'NCT04172922', 'NCT06937944', 'NCT06447480', 'NCT06798363', 'NCT06758947', 'NCT04695977', 'NCT06921850', 'NCT06833307', 'NCT06242288', 'NCT04792073', 'NCT06687967', 'NCT05932654', 'NCT06947928', 'NCT06767540', 'NCT02231775', 'NCT05983237', 'NCT06226610', 'NCT07040436', 'NCT02799485', 'NCT05789056', 'NCT06797544', 'NCT01297400', 'NCT04400994', 'NCT06647069', 'NCT06188546', 'NCT07076706', 'NCT05571943', 'NCT06035354', 'NCT06880042', 'NCT05169554', 'NCT04303169', 'NCT06365619', 'NCT06685835', 'NCT05165069', 'NCT05769777', 'NCT06945458', 'NCT04901195', 'NCT05196373', 'NCT06112314', 'NCT06857799', 'NCT06445023', 'NCT04050436', 'NCT04569409', 'NCT06630559', 'NCT04697576', 'NCT06418724', 'NCT07008547', 'NCT03011814', 'NCT05139602', 'NCT06931119', 'NCT06804811', 'NCT04331093', 'NCT05526521', 'NCT05699603', 'NCT06844799', 'NCT06358677', 'NCT06752343', 'NCT06968559', 'NCT06661382', 'NCT05725876', 'NCT05888844', 'NCT05584007', 'NCT06511973', 'NCT07007273', 'NCT06939036', 'NCT05990725', 'NCT06090266', 'NCT05377905', 'NCT03756389', 'NCT06297967', 'NCT06994520', 'NCT06917690', 'NCT07117851', 'NCT03993106', 'NCT04708418', 'NCT06555328', 'NCT06620692', 'NCT06120140', 'NCT01352520', 'NCT06018987', 'NCT04999631', 'NCT06283550', 'NCT06095102', 'NCT05393713', 'NCT06485219', 'NCT03240211', 'NCT05464381', 'NCT04642287', 'NCT06073119', 'NCT06665594', 'NCT06300502', 'NCT06444165', 'NCT06340984', 'NCT05157958', 'NCT04772079', 'NCT07042295', 'NCT05078385', 'NCT02910700']
    # total_list_nctid_200 = Gene_therapy_clinical_trials+Skin_therapy_clinical_trials
    # try_point = []
    # try_point.append(total_list_nctid_200[0])
    with open(args.nctid_that_you_want_to_input, 'rb') as f:
         nctid_list= pickle.load(f)
 
    #skin_results_after_0201 = ['NCT05634252', 'NCT06481163', 'NCT03271372', 'NCT06940895', 'NCT06388642', 'NCT06301178', 'NCT06060977', 'NCT05608187', 'NCT06386744', 'NCT05087849', 'NCT05162586', 'NCT04569409', 'NCT04708418', 'NCT06073119', 'NCT05165069']
    print(f"total nctid number is {len(nctid_list)}")
    all_sampled_urls_pairs=[]
    all_urls = set()
    for i in nctid_list:    #total_list_nctid_200 skin_results_after_0201
        # print(i)
        corrsponding_url = nctid2url[i]
        for url in corrsponding_url:
            all_urls.add(url)
            sample_tuple = (i,url)
            all_sampled_urls_pairs.append(sample_tuple)
    print(f"all urls number is {len(all_urls)}")
    all_urls = list(all_urls)


    db_urls = get_all_urls_set()


    not_in_db = set(all_urls)-set(db_urls)
    to_get = set(all_urls) - not_in_db
    url2longfilatura  = get_url_metadata_dict(list(to_get))

    # for i, k in url2longfilatura.items():
    #     print('url2longfilatura is :')
    #     print(i)
    #     print('and k is')
    #     print(k)
    #     break

    prompt = """You are an information-extraction assistant. From the clinical-trial description below, extract ONLY:
    - IDs (registry IDs like NCT/EudraCT/ISRCTN/ANZCTR/CTRI/ChiCTR/DRKS/UMIN/JPRN/WHO UTN; internal study/protocol/other IDs such as LTS16004, DRI15928, HCWPAN102).
    - Drug identifiers actually used in the trial (generic names, brand names, and molecule/product codes like SAR442168, GEN3013, HCW9218, EPKINLY).
    - Program/trial acronyms or nicknames that name this trial or its direct parent/extension (e.g., KEYNOTE-355, CheckMate 649, OLYMPIA, OLYMPIA LTE, EPCORE DLBCL-1).
    - Sponsor or collaborator organization names (e.g., Sanofi, ImmunityBio, Genmab, AbbVie).

    Exclude everything else (phases, design features, endpoints, populations, locations, enrollment, dosing, comparators, methods like “3+3”, MTD/RP2D labels, non-identifier free text).

    Rules
    1. Quote verbatim: copy exact strings (preserve case, hyphens, slashes).
    2. No guessing: include only text present; do not normalize beyond trimming whitespace.
    3. Deduplicate: if duplicates or minor variants occur, keep the most complete form once.
    4. Atomic items: each list element is a single identifier string.
    5. Parent links: if a parent/extension study is named with an ID or program acronym, include that ID/acronym; ignore any non-identifier context.
    6. Handling slashes in candidate IDs:
    - Split a slash-separated string into separate IDs ONLY when both sides represent peer study codes sharing the same alphabetic prefix or when the right-side omits the prefix but clearly inherits it from the left.
        Examples to SPLIT:
        “PAC203/PAC303” → “PAC203”, “PAC303”
        “PAC203/303” → “PAC203”, “PAC303”
    - Otherwise, KEEP the slash as part of a single identifier (do NOT split). This includes namespace-like or catalog-like IDs, or where no shared prefix is evident.
        Examples to KEEP:
        “AA/211” → “AA/211”
        “IND/BB-12345” → “IND/BB-12345”
        Any URL or path-like text → keep intact and generally exclude unless it’s explicitly an ID.
    - Never split registry IDs, URLs, or drug names due to slashes.

    Output format (must follow exactly)
    - Start a new line with LIST: then a single valid Python list of strings.
    - No extra commentary before or after the list.
    - Example:
    LIST: ["NCT01234567", "EudraCT 2019-000123-45", "LTS16004", "DRI15928", "HCWPAN102", "tolebrutinib", "SAR442168", "GEN3013", "EPKINLY", "KEYNOTE-355", "OLYMPIA", "OLYMPIA LTE", "EPCORE DLBCL-1", "Sanofi", "ImmunityBio, Inc.", "Genmab", "AbbVie"]

    Description
    {description}
    """

    # with open("../data/sep25_match_keywords.pickle","wb") as f:
    #     pickle.dump(nct2ids_300,f)
    #  ./data/sep25_all_descriptions.pickle
    with open(args.nctid_to_description,"rb") as f:
        all_descriptions = pickle.load(f)



    # with open("./data/sep25_url2longfilatura.pickle","rb") as f:
    #     url2longfilatura = pickle.load(f)


    for i in tqdm(all_descriptions):
        new_all_descriptions = set()
        for k in all_descriptions[i]:
            new_all_descriptions.add(k)
        all_descriptions[i] = new_all_descriptions


    identify_ct_instruction = """
    Given the description of a clinical trial and a scraped webpage, (if the webpage is a research paper, ignore any scraped part that is in the reference section or similar articles or external links, because they are not the main contents of the webpage, and should not be considered by you), decide if the webpage contains information about the same exact clinical trial, or at least a part of the same exact clinical trial.

    If the webpage does not reference any clinical trial, then you should output "no". Note, if a webpage contains case reports of patients, these cases could be part of a clinical trial. Even if the webpage does not explicitly mention a clinical trial but does mention case reports, this is not evidence for you to output "no".

    The webpage may reference multiple clinical trials, with different level of details provided. However, no matter how detailed each clinical trial is described, you must check all of them carefully against the given clinical trial, instead of purely focusing on the main clinical trial mentioned on the webpage.

    Note, if the webpage discusses a similar clinical trial and for this trial it explicitly gives a different NCT ID from the given clinical trial, this is not the same trial as the given trial and this cannot be a part of the given trial, therefore you should output "no". If the webpage references the clinical trial using a different study identifier that is not NCT ID, and if the given clinical trial also has another identifier that has the same format as the webpage's trial's study identifier, and if they exactly match, output "yes". Otherwise, output "no". For example, if the webpage's trial has an identifier that's SPX-017 and the given trial has an identifier that's SPX-011, then they are different trials and you should output "no". If the webpage's trial has an identifier that's ABC-017, and the given trial has an identifier that's SPX-011, because they do not have the same format, clearly they don't use the same identification convention. Then you cannot compare them. This is neither evidence for you to output "yes" nor "no".

    For both the webpage's trial, and the given clinical trial, identify and output the following information for both, NCT IDs (if available), other study identifiers (for example SPX-017, if available), treatment/intervention (e.g., drugs and procedures), study design, sponsor/investigator/collaborator/affiliated parties/organization, phase, location, enrollment (or the number of patients reported). Below is a detailed guideline of how you should use this information for both trials to determine if the webpage's trial is the same as (or a subset of) the given trial, or if they are different trials. 

    If the webpage discusses a similar trial, but the intervention treatment is not the same (e.g. different combination of drugs, different dosage, different delivery method, different control etc.) as any arm or group of the given clinical trial, then the trial cannot be the same as the given trial nor can it be a subset of the given trial, therefore you should output "no". 

    Note it is acceptable if the webpage's trial has fewer intervention treatments than the given trial, because the webpage's trial could be a subset or a few arms of the given trial. Thus, if the webpage's trial has fewer treatments, it is not evidence for you to output "no". Importantly, however, if the webpage's trial has more treatment, e.g., more drugs or more procedures, or has slightly different treatment, e.g., some different drugs or procedures, then the webpage's trial has no chance of being a subset of a few arms of the given trial. Thus, if the webpage's trial has more or different treatments than the given trial, this is evidence for you to output "no". 

    If the webpage's trial uses a different study design, such as a different masking/blinding technique, than the given clinical trial, you should output "no".

    Importantly, you must immediately identify investigator, sponsor, collaborator, affiliated parties, organization information from both the webpage's trial and the given trial, when available. If the webpage's trial has a completely distinct set of sponsor/investigator/collaborator/affiliated parties/organization from the set of information for the given trial, and you make sure there is no way this set of information from the webpage's trial overlaps with the set of information from the given trial, then you should output "no". 

    You should output "no" if any of the two following situations happens: The webpage discusses a similar trial but the location or phase information is drastically different from the given clinical trial, which means there is no way the webpage's discussed trial is the same as or a part of the given clinical trial. For example, if the given trial is phase 2 and phase 3 (i.e. the given trial has two components, one for phase 2 and the other for phase 3), but the webpage's trial is phase 1. For another example, if the given trial only has a location in San Francisco, but the webpage's trial has multiple locations, then there is no way the webpage's trial is a part of the given trial. Location information is usually city specific, therefore, two different cities in the same country does not mean the same location. You should output "no" if any of the above two situations happens.

    However, different sources can have slightly different definitions of phase information. For instance, if the given trial is phase 2/3, and the webpage's trial is explicitly mentioned to be phase 3 or phase 2a or phase 2b, it is possible they have slightly different definitions of phase, thus this is not evidence that they are not the same trial. If the given trial's location is multi-location, and the webpage's trial only reports a single location, it is possible the webpage's trial is a subset of the given trial. Therefore, you should not use these as evidence to output "no".

    If the webpage's trial has a much larger enrollment number than the given trial, then there is no way the webpage is a subset or the same as the given trial, therefore you should output "no".

    However, if the webpage's trial has a much lower enrollment number than the given trial, and you find the given trial has multiple cohorts or groups and some cohorts have approximately the same number of participants as the webpage trial's enrollment number, then it is possible the webpage is discussing a group or subset of the given trial. You should not use this as evidence to output "no".

    If the webpage's trial uses very different participants eligibility such as inclusion/exclusion criteria, or if you decide that at least one patient in the webpage's trial does not explicitly meet the inclusion criteria of the given clinical trial, or if you find that at least one patient in the webpage's trial meet the exclusion criteria of the given trial, this means there is no way this trial is the same as or a part of the given trial, then you should output "no".

    Overall, the guiding principle is: there is certain information such as enrollment number, phase information that can be different but it is still possible the webpage's trial is a subset or the same as the given trial. However, there is other information, such as study identifier, treatment, drug, study design, sponsor/investigator/collaborator/affiliated parties/organization, and eligibility criteria that cannot be different.

    To repeat the previous instruction, for both the webpage's trial, and the given clinical trial, identify and output the following information for both, NCT IDs (if available), other study identifiers (for example SPX-017, if available), treatment/intervention (e.g., drugs and procedures), study design, sponsor/investigator/collaborator/affiliated parties/organization, phase, location, enrollment (or the number of patients reported). Above is the aforementioned detailed guideline of how you should use this information for both trials to determine if the webpage's trial is the same as (or a subset of) the given trial, or if they are different trials. 

    After you have carefully compared the webpage's trial (if there is any) and the given trial, you should have decided if they are the same trial or at least the webpage is reporting part of the given trial, if you decide yes, you should output "yes". Otherwise, directly output "no".

    Clinical trial description:
    {clinical_trial_description}

    Scraped Webpage:
    {scraped_webpage}

    Output either "yes" or "no" after the keyword "OUTPUT:"
    """.strip()

    # with open("../data/sep25_sample_100_trials_keys.pickle","wb") as f:
    #     pickle.dump(sampled_100,f)


    _PARTIAL_RE = re.compile(r"^(\d{4})(?:-(\d{2})(?:-(\d{2}))?)?$")


    gpt5_prompt_dict = {}
    added_word_len = []
    splitted_contents = {}
    # added_word_len = []
    total_token_length = 0
    used_pairs = []
    enc = get_oai_encoding("gpt-4o")
    prrompt_token_len = len(enc.encode(identify_ct_instruction))
    
    for nctid,url in tqdm(all_sampled_urls_pairs):
        """
        问题在这里,我们的nctid对应的URL似乎在earliest_dates_all中找不到对应的key
        """
        #d = earliest_dates_all.get(url)   # 不存在则返回 None
        # if d is not None and is_before_with_partial("2025-02-01", d): #if earliest_dates_all[url] is not None and is_before_with_partial("2025-02-01", earliest_dates_all[url]):
            # 1) earliest_dates_all 缺失
        d = earliest_dates_all.get(url, None)
        if d is None:
            _mark_fail("missing_earliest_date", nctid, url)
            continue

        # 2) earliest_date 不满足时间条件
        if not is_before_with_partial("2025-02-01", d):
            _mark_fail("date_not_before_2025_02_01", nctid, url)
            continue

        # 3) url 不在 DB（你前面做了 to_get / not_in_db，通常这里不会发生，但稳妥起见）
        #    如果你认为“不在 DB”也算没配对，那就记录
        if url not in url2longfilatura:
            _mark_fail("missing_longfilatura_in_db", nctid, url)
            continue
        used_pairs.append((nctid,url))
        curr_ele = all_36k_trials[nctid]
        processed_ele = strip_dates_from_trial(curr_ele)
        if url in url2longfilatura:
            curr_url_content = '\n\n Actual content: ' + url2longfilatura[url]
        else:
            curr_url_content = ""
        # added_words = "Snippet: " + '\n'.join(all_descriptions[(nctid,url)]) + '\n\n Actual content: '
        # added_word_len.append(len(added_words.split(' ')))

        curr_url_content = "Snippet: " + '\n'.join(all_descriptions[(nctid,url)]) + curr_url_content
        curr_url_content = normalize_common_dashes(curr_url_content)
        enc = get_oai_encoding("gpt-4o")
        encoding_len = len(enc.encode(curr_url_content))
        total_token_length += encoding_len
        MAX_PER_CHUNK = 100_000
        if encoding_len > MAX_PER_CHUNK:
            curr_url_content_parts = split_by_tokens(curr_url_content, MAX_PER_CHUNK, enc)
            splitted_contents[(nctid, url)] = curr_url_content_parts
        else:
            curr_url_content_parts = [curr_url_content]
            splitted_contents[(nctid, url)] = curr_url_content_parts

        for index, s in enumerate(curr_url_content_parts):
            my_prompt = identify_ct_instruction.format(clinical_trial_description=processed_ele, scraped_webpage=s)
            gpt5_prompt_dict[(nctid, url, index)] = my_prompt
        #else: print(f"No valid (nctid, url) pairs found before 2025-02-01.{url} with date {d}")
    
    len(gpt5_prompt_dict)
        



    # for k, v in gpt5_prompt_dict.items():
    #     print(f"Key: {k}")
    #     print(v)
    #     print("-" * 40)
    #     break


    # do not use the following code until you need to load new file or data

    with open(f"{args.output_dir_to_round1_before_0201}/{args.exp_name}_round1_prompt.pickle","wb") as f:
        pickle.dump(gpt5_prompt_dict,f)

    print("prompt dict len: ", len(gpt5_prompt_dict))

    # gpt5_reference_ret_long, gpt5_timeout_dict, _ = generate(
    #     prompt_dict=gpt5_prompt_dict,
    #     model='gpt-5-2025-08-07',
    #     max_completion_tokens=10000,
    #     reasoning='medium',
    #     verbosity='low',
    #     timeout=360,
    # )

    # # don't use it until you load new sample or data
    # with open(f"./gpt_responses/{args.exp_name}_round1_responses_before_02_01.pickle","wb") as f:
    #     pickle.dump(gpt5_reference_ret_long,f)



    # remember data to use it in round2
    with open(f"{args.output_dir_to_round1_before_0201}/{args.exp_name}_splitted_contents.pickle","wb") as f:
        pickle.dump(splitted_contents,f)
    with open(f"{args.output_dir_to_round1_before_0201}/{args.exp_name}_used_pairs.pickle","wb") as f:
        pickle.dump(used_pairs,f)
    
    with open(f"{args.output_dir_to_round1_before_0201}/{args.exp_name}_failed_pairs.pickle", "wb") as f:
        pickle.dump(failed_pairs, f)

    print("===== FAILED SUMMARY =====")
    print("failed total:", len(failed_pairs))
    for k, v in sorted(failed_reason.items(), key=lambda x: -x[1]):
        print(f"{k}: {v}")
    print("saved failed pairs to:", f"{args.output_dir_to_round1_before_0201}/{args.exp_name}_failed_pairs.pickle")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="pipeline match by gpt before 0201 round 1.")
    parser.add_argument('--exp_name', type=str, required=True,  help='path to save the gpt match round1 before 0201 results.')
    parser.add_argument('--earliest_dates_path', type=str, required=True, help='path to load the file of earliest_dates_all.')
    parser.add_argument('--nctid_to_url_path', type=str, required=True, help='path to load the file of nctid and relevant url.')
    parser.add_argument('--all_36k_trials_path', type=str, required=True, help='path to load the file of all 36k clinical trials.')
    parser.add_argument('--nctid_to_description', type=str, required=True, help='path to load the file of nctid and relevant description.')
    parser.add_argument('--nctid_that_you_want_to_input', type=str, required=True, help='the pickle that contains the nctid you want to input.')
    parser.add_argument('--output_dir_to_round1_before_0201', type=str, required=True, help='directory to save the data that will be used in round2_before_0201.')
 
    # Add any arguments you want to parse here
    args = parser.parse_args()
    main(args)


