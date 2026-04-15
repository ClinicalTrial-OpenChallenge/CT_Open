import psycopg2
import psycopg2.extras
import pickle
from tqdm import tqdm
from parse_html_page_multi_thread import parse_html_page_parallel
from parse_html_page_utils import normalize_to_url_html_map
from generation import generate
from get_html_date import find_date_html_multithreaded, find_date_html
from datetime import datetime
import time
MAX_WORKERS = 16

# INSERT_QUERY = """
# INSERT INTO scrapes (url, raw, longfilatura, pub_date, scrape_type, searchapi)
# VALUES %s
# ON CONFLICT (url) DO NOTHING;"""
INSERT_QUERY = """
INSERT INTO scrapes (url, raw, longfilatura, pub_date, scrape_type, searchapi)
VALUES %s
ON CONFLICT (url) DO UPDATE SET
    raw = EXCLUDED.raw,
    longfilatura = EXCLUDED.longfilatura,
    pub_date = EXCLUDED.pub_date,
    scrape_type = EXCLUDED.scrape_type,
    searchapi = EXCLUDED.searchapi;
"""
# import unicodedata
# from collections.abc import Mapping, Iterable

# def _strip_surrogates(s: str, replace_with=None) -> str:
#     # remove or replace U+D800..U+DFFF
#     if replace_with is None:
#         return ''.join(ch for ch in s if not (0xD800 <= ord(ch) <= 0xDFFF))
#     return ''.join(ch if not (0xD800 <= ord(ch) <= 0xDFFF) else replace_with for ch in s)

# def sanitize_text(s: str) -> str:
#     # Remove NULs, strip surrogates, and normalize
#     s = s.replace('\x00', '')
#     s = _strip_surrogates(s, replace_with='')   # or '�' if you prefer marking the spot
#     try:
#         s = unicodedata.normalize('NFC', s)
#     except Exception:
#         pass
#     return s

# def sanitize_value(x):
#     if x is None:
#         return None
#     if isinstance(x, str):
#         return sanitize_text(x)
#     if isinstance(x, (bytes, bytearray)):
#         # If destined for BYTEA, wrap with psycopg2.Binary(x) instead of converting.
#         # If destined for TEXT and you truly don't know the encoding, this makes a lossy but valid string:
#         return x.decode('utf-8', 'replace')
#     return x

# fast_sanitize.py
import unicodedata
from functools import lru_cache

# Build once: delete NUL (0) and all surrogates U+D800..U+DFFF
_TRANSLATE_DELETE = {0: None}
_TRANSLATE_DELETE.update({cp: None for cp in range(0xD800, 0xE000)})

@lru_cache(maxsize=20000)
def _sanitize_small(s: str) -> str:
    # Caches small, often-repeated values (statuses, country codes, etc.)
    return s.translate(_TRANSLATE_DELETE)

def sanitize_text(s: str, *, normalize: bool = False) -> str:
    # Fast path for common short strings
    if len(s) <= 512:
        out = _sanitize_small(s)
    else:
        out = s.translate(_TRANSLATE_DELETE)

    if normalize:
        # Only normalize if necessary (Python 3.10 has is_normalized)
        try:
            if not unicodedata.is_normalized("NFC", out):
                out = unicodedata.normalize("NFC", out)
        except AttributeError:
            # Fallback for very old Pythons
            out = unicodedata.normalize("NFC", out)
    return out

def sanitize_row(row, *, normalize: bool = False, binary_cols=()):
    """
    Fast sanitizer for a single DB row (tuple/list).
    - Strings: delete NULs + surrogates (and optionally NFC-normalize).
    - Bytes in `binary_cols`: wrap with psycopg2.Binary to keep as BYTEA.
    - Other bytes: decode with UTF-8 'replace' (lossy but valid TEXT).
    - Everything else: passthrough.
    """
    # Local bindings for speed
    _san = sanitize_text
    _Binary = None
    if binary_cols:
        # Import only if needed, avoids import cost on hot path
        from psycopg2 import Binary as _Binary

    out = []
    for idx, v in enumerate(row):
        if isinstance(v, str):
            out.append(_san(v, normalize=normalize))
        elif isinstance(v, (bytes, bytearray)):
            if idx in binary_cols:
                out.append(_Binary(v) if _Binary else v)  # BYTEA
            else:
                out.append(v.decode("utf-8", "replace"))  # TEXT-safe
        else:
            out.append(v)
    return tuple(out)

def sanitize_rows(rows, *, normalize: bool = False, binary_cols=()):
    # Generator to avoid materializing a huge list at once
    for row in rows:
        yield sanitize_row(row, normalize=normalize, binary_cols=binary_cols)


def parse_html_page(url2raw, response_file_name):
        # The initial parallel parsing step remains the same.
    _, url2my_dict, url2_processed_results_list = parse_html_page_parallel(url2raw, MAX_WORKERS)

    # This loop contains the filtering block you wanted to keep.
    for url in tqdm(url2my_dict):
        my_dict, filtered_results_list = url2my_dict[url]
        if my_dict is not None:
            assert 'ackn' not in my_dict
            all_counts = [tup[0] for k, tup in my_dict.items()]
            flag = False
            if sum(all_counts) == 0:
                flag = True

            indicator = [1 for k in my_dict if my_dict[k][0] == 1]
            if sum(indicator) > 0:
                flag = True

            # The 'if not flag' block's structure is preserved.
            if not flag:
                # The logic that used 'decisions' to find 'all_yes_indices'
                # and truncate the list has been removed from this block.
                # As a result, this block no longer alters 'filtered_results_list'.
                pass

        # The list is updated (in this case, reassigned without modification).
        url2_processed_results_list[url] = filtered_results_list

    # The final cleaning and string-joining logic remains.
    cleaned_url2_processed = {}
    for k in url2_processed_results_list:
        ind_ele = [i[0].strip() for i in url2_processed_results_list[k]]
        res_str = ' '.join(ind_ele)
        cleaned_url2_processed[k] = res_str

    return cleaned_url2_processed

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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data_path', type=str,required=True, help='input datapath')
    # parser.add_argument('--num_batches', type=int,required=True, help='number of batches')
    args = parser.parse_args()

    base_path = args.input_data_path
    # with open("./pipeline_data/sep18_used_files_insert_db.pickle",'rb') as f:
    #     used_files = pickle.load(f)
    # used_files = set(used_files)
    # if base_path in used_files:
    #     print(f"File {base_path} already processed, skipping")
    #     exit()
    # base_path = "../scraping/jul20_scraped_results_batch_{}.pickle"
    # for i in tqdm(range(1, args.num_batches + 1)):
    print(f"\nProcessing input file {base_path}")
    with open(base_path, "rb") as f:
        data = pickle.load(f)
    if "scraped_results_batch" not in data or "batch_skipped_by_extension" not in data:
        print(f"Data {base_path} does not have scraped_results_batch, not a scraped url file")
        exit()
    # if not data:
    #     continue
    ####
    # Comment: need to drop this column after all insertions, since all the urls are mixed together to run the pipeline
    ####
    if "serp" in base_path.lower():
        searchAPI = 'serpapi'
    else:
        searchAPI = 'brave'
    print(f'Data {base_path} loaded')

    url2raw_raw = normalize_to_url_html_map(data)
    url2raw = {}
    short_filename = base_path.split('/')[-1].replace('.pickle', '_too_long_urls.pickle')
    too_long_urls = set()

    database_urls = set(get_all_urls_set())
    print(f"Found {len(database_urls)} URLs in the database")
    curr_urls = set(url2raw_raw.keys())
    print(f"Found {len(curr_urls)} URLs in the current file")
    run_urls = curr_urls
    print(f"Found {len(run_urls)} URLs to process")
    for url in tqdm(run_urls, desc="Checking too long urls"):
        words_len = len(url2raw_raw[url].split(" "))
        if words_len > 800000:
            too_long_urls.add((base_path, url))
        else:
            url2raw[url] = url2raw_raw[url]
    del url2raw_raw

    with open(f"./too_long_urls/{short_filename}",'wb') as f:
        pickle.dump(too_long_urls, f)

    print(f"Found {len(url2raw)} URLs to process")
    ###
    # Comment: Date the html page as accurate as possible
    ###
    print("Running find_date_html_multithreaded")
    url2pub_date = find_date_html_multithreaded(url2raw, MAX_WORKERS)
    # url2pub_date = find_date_html(url2raw)


    ###
    # Comment: Process the html page and keep only the onces needed
    ###
    print("Running parse_html_page")
    url2_processed_results_list = parse_html_page(url2raw, base_path.split('/')[-1].replace('.pickle', '_gpt_response.pickle'))


    pubdate_urls = set(url2pub_date.keys())
    processed_urls = set(url2_processed_results_list.keys())
    final_urls = pubdate_urls.intersection(processed_urls)
    # Prepare the list of tuples for the current file.
    data_to_insert = [
        (k, v['content'], url2_processed_results_list[k],
            url2pub_date[k], v['method'], searchAPI)
        for k, v in tqdm(data['scraped_results_batch'].items(), desc="Preparing data to insert") if k in final_urls
    ]
    print(f'Starting to insert data {len(data_to_insert)}')
    print("Start time: ", datetime.now())
    start_time = time.time()

    # with open("data_to_insert.pickle", "wb") as f:
    #     pickle.dump(data_to_insert, f)
    with psycopg2.connect(
        dbname="web_database",
        user="postgres",
        host="/tmp"
    ) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cur:
            if data_to_insert:
                psycopg2.extras.execute_values(
                    cur,
                    INSERT_QUERY,
                    sanitize_rows(data_to_insert, normalize=False, binary_cols=()),
                    page_size=1000
                )
                inserted_count = cur.rowcount  # Number of inserted rows
                # print(f"Upsert complete. {inserted_count} rows inserted.")
                # print(f"Conflicted rows: {len(data_to_insert) - inserted_count}")
                conn.commit()
    end_time = time.time()
    print("End time: ", datetime.now())
    print(f"Time taken to insert: {end_time - start_time} seconds")

    # with open("./pipeline_data/aug27_used_files_insert_db.pickle",'wb') as f:
    #     pickle.dump(used_files,f)
    print(f"File {base_path} inserted successfully.")
    # print(f"Batch file {} inserted successfully.")
