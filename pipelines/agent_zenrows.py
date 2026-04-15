import sys
import os
import concurrent.futures
import psycopg2
import tiktoken
import re
from zenrows import ZenRowsClient
from tqdm import tqdm
from urllib.parse import urlparse
from datetime import date
import os
# Add paths for utility modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))
sys.path.insert(0, os.path.dirname(__file__))

from get_pubmed_pmc_paper_date import build_url2paper_date
from get_html_date import find_date_html_multithreaded, find_date_html
from parse_html_page_multi_thread import parse_html_page_parallel
from parse_html_page import parse_html_page
import unicodedata
from functools import lru_cache

# --- Configuration ---
ZENROWS_API_KEY = os.environ.get("ZENROWS_API_KEY")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
MIN_TOKEN_THRESHOLD = 50
MAX_WORKERS = 16

DOWNLOADABLE_EXTENSIONS = {
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.odt', '.ods', '.odp',
    '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
    '.mp3', '.wav', '.mp4', '.mov', '.avi', '.mkv', '.flv',
    '.csv', '.json', '.xml', '.js', '.css', '.iso',
}

# --- Sanitization (mirrors insert_to_db.py) ---
_TRANSLATE_DELETE = {0: None}
_TRANSLATE_DELETE.update({cp: None for cp in range(0xD800, 0xE000)})

@lru_cache(maxsize=20000)
def _sanitize_small(s: str) -> str:
    return s.translate(_TRANSLATE_DELETE)

def _sanitize_text(s: str) -> str:
    if len(s) <= 512:
        return _sanitize_small(s)
    return s.translate(_TRANSLATE_DELETE)

def _sanitize_row(row):
    out = []
    for v in row:
        if isinstance(v, str):
            out.append(_sanitize_text(v))
        elif isinstance(v, (bytes, bytearray)):
            out.append(v.decode("utf-8", "replace"))
        else:
            out.append(v)
    return tuple(out)

_INSERT_QUERY = """
INSERT INTO scrapes (url, raw, longfilatura, pub_date, scrape_type, searchapi)
VALUES %s
ON CONFLICT (url) DO UPDATE SET
    raw = EXCLUDED.raw,
    longfilatura = EXCLUDED.longfilatura,
    pub_date = EXCLUDED.pub_date,
    scrape_type = EXCLUDED.scrape_type,
    searchapi = EXCLUDED.searchapi;
"""

def _insert_to_db(rows):
    """
    Insert rows into the scrapes table.

    Each row is a tuple: (url, raw, longfilatura, pub_date, scrape_type, searchapi).
    - For papers: raw = paper_text, longfilatura = paper_text, pub_date = paper_date.
    - For HTML:   raw = raw_html,   longfilatura = cleaned_text, pub_date = extracted date.
    """
    if not rows:
        return
    import psycopg2
    import psycopg2.extras
    sanitized = [_sanitize_row(r) for r in rows]
    try:
        with psycopg2.connect(dbname="web_database", user="postgres", host="/tmp") as conn:
            with conn.cursor() as cur:
                psycopg2.extras.execute_values(cur, _INSERT_QUERY, sanitized, page_size=100)
                conn.commit()
        print(f"  DB insert: {len(sanitized)} row(s) inserted/updated.")
    except Exception as e:
        print(f"  DB insert failed: {e}")


def _get_encoding():
    try:
        return tiktoken.encoding_for_model("gpt-4o")
    except Exception:
        try:
            return tiktoken.get_encoding("o200k_base")
        except Exception:
            return tiktoken.get_encoding("cl100k_base")


def is_downloadable_file_by_extension(url):
    try:
        parsed_path = urlparse(url).path
        if parsed_path.lower().endswith(tuple(DOWNLOADABLE_EXTENSIONS)):
            return True
    except Exception:
        return False
    return False


def _is_database_article(url):
    """Check if URL is a PubMed/PMC/bioRxiv/medRxiv database article."""
    return '/PMC' in url or 'pubmed' in url.lower() or 'medrxiv' in url.lower() or 'biorxiv' in url.lower()


def _is_date_before_cutoff(date_str, cutoff):
    """
    Check if a partial date string (YYYY, YYYY-MM, or YYYY-MM-DD) is strictly
    before the cutoff date.
    - None or unparseable -> returns False (EXCLUDE: unknown date is not trusted).
    - Year-only like '2025' -> returns False (EXCLUDE: ambiguous, could be after cutoff).
    - YYYY-MM or YYYY-MM-DD -> compare against cutoff using latest possible day.

    Args:
        date_str: Date string in YYYY, YYYY-MM, or YYYY-MM-DD format.
        cutoff: Cutoff date string in YYYY-MM-DD format (e.g. "2025-01-31").
    """
    if not date_str:
        return False  # no date -> exclude
    # Parse cutoff
    cutoff_parts = cutoff.strip().split("-")
    cutoff_date = date(int(cutoff_parts[0]), int(cutoff_parts[1]), int(cutoff_parts[2]))
    parts = date_str.strip().split("-")
    try:
        year = int(parts[0])
        if len(parts) < 2:
            return False  # year-only -> exclude (ambiguous)
        month = int(parts[1])
        day = int(parts[2]) if len(parts) >= 3 else 28  # conservative upper bound
        month = min(max(month, 1), 12)
        day = min(max(day, 1), 28)
        d = date(year, month, day)
        return d < cutoff_date
    except (ValueError, IndexError):
        return False  # can't parse -> exclude


def get_url_metadata_dict(
    urls,
    *,
    dbname="web_database",
    user="postgres",
    password=None,
    host="/tmp",
    port=None,
    batch_size=10_000,
    fetch_size=10_000,
):
    """
    For a given list of URLs, return a dict: {url: (raw_content_or_None, pub_date_or_None)}.
    Retrieves the 'raw' and 'pub_date' columns from the scrapes table.
    """
    url_list = list(dict.fromkeys(u for u in urls if u))
    result = {u: (None, None) for u in url_list}
    if not url_list:
        return result

    connect_kwargs = dict(dbname=dbname, user=user, host=host)
    if password:
        connect_kwargs["password"] = password
    if port:
        connect_kwargs["port"] = port

    conn = psycopg2.connect(**connect_kwargs)
    try:
        conn.set_session(readonly=True, autocommit=False)
        for i in range(0, len(url_list), batch_size):
            batch = url_list[i : i + batch_size]
            cursor_name = f"meta_stream_{i // batch_size}"
            with conn.cursor(name=cursor_name) as cur:
                cur.itersize = fetch_size
                cur.execute(
                    "SELECT url, raw, pub_date FROM scrapes WHERE url = ANY(%s)",
                    (batch,),
                )
                for u, raw, pub_date in cur:
                    # pub_date may be a date/datetime object; convert to string
                    pub_date_str = None
                    if pub_date is not None:
                        pub_date_str = str(pub_date) if not isinstance(pub_date, str) else pub_date
                    result[u] = (raw, pub_date_str)
        conn.rollback()
    finally:
        conn.close()
    return result


def _scrape_single_url(url):
    """Scrape a single URL using ZenRows premium. Returns (url, content, method)."""
    try:
        client = ZenRowsClient(ZENROWS_API_KEY)
        params = {
            'antibot': 'true',
            'js_render': 'true',
            'premium_proxy': 'true',
            'wait_for': 'body',
            'proxy_country': 'us',
        }
        response = client.get(url, params=params, timeout=90)
        response.raise_for_status()
        return url, response.text, 'zenrows_premium'
    except Exception as e:
        return url, f"Failed with ZenRows Premium: {e}", 'failed'


def _clean_html_content(url2raw):
    """
    Run parse_html_page_parallel on a url->raw_html dict and return
    cleaned {url: cleaned_text}. Mirrors the parse_html_page function
    in insert_to_db.py.
    """
    if not url2raw:
        return {}

    _, url2my_dict, url2_processed_results_list = parse_html_page_parallel(
        url2raw, max_workers=MAX_WORKERS
    )

    # Apply the same filtering logic as insert_to_db.py
    for url in url2my_dict:
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
            if not flag:
                pass
        url2_processed_results_list[url] = filtered_results_list

    # Join cleaned text segments
    cleaned = {}
    for k in url2_processed_results_list:
        ind_ele = [i[0].strip() for i in url2_processed_results_list[k]]
        res_str = ' '.join(ind_ele)
        cleaned[k] = res_str

    return cleaned


def scrape_one_url(url, date_cutoff):
    """
    Scrape a single URL and return a dict {url: content} if it passes date
    filtering, or an empty dict otherwise.

    Same logic as scrape_urls but without any multithreading.

    Args:
        url: A single URL string.
        date_cutoff: Cutoff date string in YYYY-MM-DD format (e.g. "2025-01-31").
                     No default -- must be provided.

    Returns:
        dict: {url: cleaned_content_string} if the URL passes all filters,
              otherwise {}.
    """
    enc = _get_encoding()

    # --- Step 1: Check database for existing raw content ---
    print(f"Step 1: Checking database for {url}...")
    db_metadata = get_url_metadata_dict([url])
    raw, pub_date_str = db_metadata.get(url, (None, None))
    has_db_content = raw is not None and len(enc.encode(raw)) > MIN_TOKEN_THRESHOLD

    # --- Step 2: Route based on URL type ---

    # 2a: Downloadable file -> skip entirely
    if is_downloadable_file_by_extension(url):
        print(f"  Skipped: downloadable file extension.")
        return {}

    # 2b: Paper URL (PubMed/PMC/bioRxiv/medRxiv)
    if _is_database_article(url):
        if has_db_content:
            # Already in DB -- just check pub_date from DB directly
            print(f"  DB hit (paper). pub_date={pub_date_str}")
            if _is_date_before_cutoff(pub_date_str, date_cutoff):
                print(f"  Kept (date < {date_cutoff}).")
                return {url: raw}
            else:
                print(f"  Filtered out (date >= {date_cutoff} / None / year-only).")
                return {}
        else:
            # Not in DB -- resolve via API
            print(f"  Paper URL not in DB. Resolving via API...")
            paper_date_map = build_url2paper_date(
                urls=[url],
                api_key=NCBI_API_KEY,
                max_workers=1,
            )
            info = paper_date_map.get(url, {})
            paper_date = info.get("date")
            paper_text = info.get("paper", "")

            if not _is_date_before_cutoff(paper_date, date_cutoff):
                print(f"  Filtered out (date={paper_date}, cutoff={date_cutoff}).")
                return {}
            if paper_text and paper_text != "UNRESOLVED":
                print(f"  Kept (date < {date_cutoff}).")
                # Insert paper to DB: raw=paper_text, longfilatura=paper_text
                _insert_to_db([(url, paper_text, paper_text, paper_date, 'api', 'agent')])
                return {url: paper_text}
            print(f"  Paper text unresolved.")
            return {}

    # 2c: Regular HTML URL
    # Get raw HTML: from DB or scrape with ZenRows
    if has_db_content:
        print(f"  DB hit (HTML page).")
        raw_html = raw
    else:
        print(f"  Scraping with ZenRows...")
        _, content, method = _scrape_single_url(url)
        if method == 'failed':
            print(f"  ZenRows failed.")
            return {}
        print(f"  ZenRows success ({method}).")
        raw_html = content

    # Extract date from HTML (single-threaded)
    url2raw = {url: raw_html}
    url2pub_date, _ = find_date_html(url2raw)
    pub_date = url2pub_date.get(url)

    if not _is_date_before_cutoff(pub_date, date_cutoff):
        print(f"  Filtered out (date={pub_date}, cutoff={date_cutoff}).")
        return {}

    print(f"  Date OK ({pub_date} < {date_cutoff}). Cleaning HTML...")

    # Clean HTML (single-threaded via parse_html_page)
    _, url2my_dict, url2_processed_results_list = parse_html_page(url2raw, response_file_name=None)

    # Apply same filtering logic as _clean_html_content
    for u in url2my_dict:
        my_dict, filtered_results_list = url2my_dict[u]
        if my_dict is not None:
            assert 'ackn' not in my_dict
            all_counts = [tup[0] for k, tup in my_dict.items()]
            flag = False
            if sum(all_counts) == 0:
                flag = True
            indicator = [1 for k in my_dict if my_dict[k][0] == 1]
            if sum(indicator) > 0:
                flag = True
            if not flag:
                pass
        url2_processed_results_list[u] = filtered_results_list

    if url in url2_processed_results_list:
        ind_ele = [i[0].strip() for i in url2_processed_results_list[url]]
        cleaned_text = ' '.join(ind_ele)
        if cleaned_text and cleaned_text.strip():
            print(f"  Done. Content length: {len(cleaned_text)} chars.")
            # Insert HTML to DB: raw=raw_html, longfilatura=cleaned_text
            if not has_db_content:
                _insert_to_db([(url, raw_html, cleaned_text, pub_date, 'zenrows_premium', 'agent')])
            return {url: cleaned_text}

    print(f"  No content after cleaning.")
    return {}


def scrape_urls(urls, date_cutoff, max_workers=60):
    """
    Scrape a list of URLs and return a dict mapping url -> content.

    Pipeline:
    1. Check the database first -- if raw content exists and is >50 tokens, use it
       directly (skip scraping). For paper URLs, also retrieve pub_date from DB.
    2. Skip downloadable files (by extension).
    3a. DB-hit paper URLs: check pub_date from DB directly against cutoff.
        No API call needed.
    3b. New paper URLs (not in DB): fetch via build_url2paper_date, filter by date.
    4. All other URLs are scraped via ZenRows premium proxy.
    5. For all HTML content (from DB + ZenRows), run find_date_html_multithreaded
       to extract dates, filter out >= date_cutoff, then clean with
       parse_html_page_parallel.

    Date filtering: URLs/papers with date >= date_cutoff, None date, or year-only
    date are ALL excluded.

    Args:
        urls: List of URL strings to scrape.
        date_cutoff: Cutoff date string in YYYY-MM-DD format (e.g. "2025-01-31").
                     No default -- must be provided.
        max_workers: Number of concurrent threads for ZenRows (default 60).

    Returns:
        dict: {url: cleaned_content_string} for URLs that pass date filtering.
              Only URLs with YYYY-MM or YYYY-MM-DD date < date_cutoff are included.
    """
    url2content = {}
    enc = _get_encoding()

    # =========================================================================
    # Step 1: Check database for existing raw content
    # =========================================================================
    print(f"Step 1: Checking database for existing raw content for {len(urls)} URLs...")
    db_metadata = get_url_metadata_dict(urls)

    db_hit_html = {}       # url -> raw HTML (from DB, needs date check + cleaning)
    db_hit_papers = {}     # url -> (raw, pub_date_str) for paper URLs in DB
    urls_needing_scrape = []

    for url in urls:
        raw, pub_date_str = db_metadata.get(url, (None, None))
        if raw is not None and len(enc.encode(raw)) > MIN_TOKEN_THRESHOLD:
            # DB has sufficient raw content -- skip scraping
            if _is_database_article(url):
                # Paper URL in DB: use pub_date directly, no need for API call
                db_hit_papers[url] = (raw, pub_date_str)
            else:
                db_hit_html[url] = raw
        else:
            urls_needing_scrape.append(url)

    print(f"  DB hits (HTML pages): {len(db_hit_html)}")
    print(f"  DB hits (paper URLs): {len(db_hit_papers)}")
    print(f"  Remaining URLs needing work: {len(urls_needing_scrape)}")

    # =========================================================================
    # Step 2: Categorize remaining URLs
    # =========================================================================
    print(f"\nStep 2: Categorizing {len(urls_needing_scrape)} remaining URLs...")
    paper_urls = []   # PubMed/PMC/bioRxiv/medRxiv not in DB
    zenrows_urls = []
    skipped_count = 0

    for url in urls_needing_scrape:
        if is_downloadable_file_by_extension(url):
            skipped_count += 1
        elif _is_database_article(url):
            paper_urls.append(url)
        else:
            zenrows_urls.append(url)

    print(f"  Skipped by file extension: {skipped_count}")
    print(f"  Paper URLs to fetch via APIs: {len(paper_urls)}")
    print(f"  URLs to scrape with ZenRows: {len(zenrows_urls)}")

    # =========================================================================
    # Step 3a: DB-hit paper URLs -- use pub_date from DB directly, skip API
    # =========================================================================
    papers_kept = 0
    papers_filtered = 0
    if db_hit_papers:
        print(f"\nStep 3a: Checking {len(db_hit_papers)} DB-hit paper URLs by pub_date...")
        for url, (raw, pub_date_str) in db_hit_papers.items():
            if not _is_date_before_cutoff(pub_date_str, date_cutoff):
                papers_filtered += 1
                continue
            url2content[url] = raw
            papers_kept += 1
        print(f"  DB papers kept (date < {date_cutoff}): {papers_kept}")
        print(f"  DB papers filtered out (date >= {date_cutoff} / None / year-only): {papers_filtered}")
    else:
        print("\nStep 3a: No DB-hit paper URLs.")

    # =========================================================================
    # Step 3b: New paper URLs (not in DB) -- fetch via NCBI/preprint APIs
    #          Filter out papers with date >= date_cutoff
    # =========================================================================
    if paper_urls:
        print(f"\nStep 3b: Resolving {len(paper_urls)} new paper URLs via APIs...")
        paper_date_map = build_url2paper_date(
            urls=paper_urls,
            api_key=NCBI_API_KEY,
            max_workers=min(max_workers, 8),
        )
        api_kept = 0
        api_filtered = 0
        paper_rows_to_insert = []
        for url in paper_urls:
            info = paper_date_map.get(url, {})
            paper_date = info.get("date")
            paper_text = info.get("paper", "")

            if not _is_date_before_cutoff(paper_date, date_cutoff):
                api_filtered += 1
                continue  # date >= cutoff / None / year-only, skip

            if paper_text and paper_text != "UNRESOLVED":
                url2content[url] = paper_text
                api_kept += 1
                # Insert paper to DB: raw=paper_text, longfilatura=paper_text
                paper_rows_to_insert.append((url, paper_text, paper_text, paper_date, 'api', 'agent'))

        _insert_to_db(paper_rows_to_insert)
        papers_kept += api_kept
        papers_filtered += api_filtered
        print(f"  API papers kept (date < {date_cutoff}): {api_kept}")
        print(f"  API papers filtered out (date >= {date_cutoff} / None / year-only): {api_filtered}")
    else:
        print("\nStep 3b: No new paper URLs to resolve.")

    print(f"  Total papers kept: {papers_kept}, filtered: {papers_filtered}")

    # =========================================================================
    # Step 4: Scrape remaining URLs with ZenRows
    # =========================================================================
    zenrows_html = {}
    if zenrows_urls:
        print(f"\nStep 4: Scraping {len(zenrows_urls)} URLs with ZenRows...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(_scrape_single_url, url): url
                for url in zenrows_urls
            }
            for future in tqdm(
                concurrent.futures.as_completed(future_to_url),
                total=len(zenrows_urls),
                desc="Scraping with ZenRows",
            ):
                url, content, method = future.result()
                if method != 'failed':
                    zenrows_html[url] = content

        print(f"  ZenRows success: {len(zenrows_html)}")
        print(f"  ZenRows failed: {len(zenrows_urls) - len(zenrows_html)}")
    else:
        print("\nStep 4: No URLs to scrape with ZenRows.")

    # =========================================================================
    # Step 5: Date filtering + content cleaning for all HTML
    #         (DB-hit HTML pages + ZenRows-scraped pages)
    # =========================================================================
    all_html = {**db_hit_html, **zenrows_html}
    if all_html:
        print(f"\nStep 5: Date filtering and cleaning {len(all_html)} HTML pages...")

        # 5a: Extract dates from HTML
        print("  Running find_date_html_multithreaded...")
        url2pub_date = find_date_html_multithreaded(all_html, max_workers=MAX_WORKERS)

        # 5b: Filter out pages with date >= date_cutoff
        html_after_date_filter = {}
        date_filtered_count = 0
        for url, html in all_html.items():
            pub_date = url2pub_date.get(url)
            if not _is_date_before_cutoff(pub_date, date_cutoff):
                date_filtered_count += 1
                continue
            html_after_date_filter[url] = html

        print(f"  Kept after date filter: {len(html_after_date_filter)}")
        print(f"  Filtered out (date >= {date_cutoff}): {date_filtered_count}")

        # 5c: Clean HTML content with parse_html_page_parallel
        if html_after_date_filter:
            print("  Running parse_html_page_parallel...")
            cleaned_html = _clean_html_content(html_after_date_filter)

            html_rows_to_insert = []
            for url, cleaned_text in cleaned_html.items():
                if cleaned_text and cleaned_text.strip():
                    url2content[url] = cleaned_text
                    # Only insert URLs that were newly scraped (not already in DB)
                    if url in zenrows_html:
                        pub_date = url2pub_date.get(url)
                        html_rows_to_insert.append(
                            (url, zenrows_html[url], cleaned_text, pub_date, 'zenrows_premium', 'agent')
                        )
            _insert_to_db(html_rows_to_insert)
            print(f"  Cleaned pages added: {len(cleaned_html)}")
    else:
        print("\nStep 5: No HTML pages to process.")

    # =========================================================================
    # Summary
    # =========================================================================
    zenrows_scraped_count = len(zenrows_html)
    successful = sum(1 for v in url2content.values() if v)
    print(f"\n{'='*50}")
    print(f"DONE. {successful}/{len(urls)} URLs returned with content.")
    print(f"  Paper URLs: {len([u for u in url2content if _is_database_article(u)])}")
    print(f"  HTML pages: {len([u for u in url2content if not _is_database_article(u)])}")
    print(f"  URLs scraped with ZenRows: {zenrows_scraped_count}")
    print(f"{'='*50}")
    return url2content, zenrows_scraped_count


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Scrape URLs with ZenRows, date filtering, and DB caching.")
    parser.add_argument("--input_path", type=str, required=True,
                        help="Path to a pickle file containing a Python list of URL strings.")
    parser.add_argument("--date_cutoff", type=str, required=True,
                        help="Cutoff date in YYYY-MM-DD format (e.g. 2025-01-31). "
                             "Only content dated strictly before this is kept.")
    parser.add_argument("--output_path", type=str, default=None,
                        help="Path to save the output pickle (url->content dict). "
                             "Defaults to <input_path_stem>_scraped.pickle.")
    args = parser.parse_args()

    with open(args.input_path, "rb") as f:
        url_list = pickle.load(f)

    assert isinstance(url_list, list), f"Expected a list of URLs, got {type(url_list)}"
    print(f"Loaded {len(url_list)} URLs from {args.input_path}")
    print(f"Date cutoff: {args.date_cutoff}")

    result = scrape_urls(url_list, date_cutoff=args.date_cutoff)

    output_path = args.output_path
    if output_path is None:
        stem = os.path.splitext(args.input_path)[0]
        output_path = f"{stem}_scraped.pickle"

    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved {len(result)} url->content entries to {output_path}")
