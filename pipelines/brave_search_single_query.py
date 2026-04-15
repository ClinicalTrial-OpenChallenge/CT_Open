import requests
import time


class BravePaymentError(Exception):
    """Raised when Brave API returns 402 Payment Required."""
    pass


API_TOKEN = "BSAhI5DS7ssaCqlX2TDJS5NWFEI7_lP"


def brave_search(query, freshness="2025-02-01", num_pages=5):
    """
    Search Brave for a query across multiple page offsets and return merged results.

    Args:
        query (str): The search query string.
        freshness (str, optional): Cutoff date in 'YYYY-MM-DD' format.
            If provided, only results up to this date are returned.
            Uses Brave's freshness parameter format: e.g. '2024-01-01to2025-06-15'.
            You can pass a single date (used as the end date) or a full range 'START_DATEtoEND_DATE'.
        num_pages (int, optional): Number of pages (offsets) to search. Defaults to 5.

    Returns:
        list[dict]: A list of dicts with keys 'title', 'description', 'url'.
    """
    all_offset_results = []
    query = query.replace('"', "")
    for offset in range(num_pages):
        params = {
            "q":              query,
            "country":        "ALL",
            "safesearch":     "off",
            "spellcheck":     "false",
            "extra_snippets": "true",
            "offset":         offset,
            "search_lang":    "en",
            "ui_lang":        "en-US",
        }
        if freshness is not None:
            params['freshness'] = freshness
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "Accept":               "application/json",
                    "Accept-Encoding":      "gzip",
                    "x-subscription-token": API_TOKEN,
                },
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            all_offset_results.append(resp.json())

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 402:
                raise BravePaymentError("Brave API returned 402 Payment Required.")
            print('sleeping for 5 seconds...')
            time.sleep(5)
            # Retry once on failure
            try:
                resp = requests.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    headers={
                        "Accept":               "application/json",
                        "Accept-Encoding":      "gzip",
                        "x-subscription-token": API_TOKEN,
                    },
                    params=params,
                    timeout=30,
                )
                resp.raise_for_status()
                all_offset_results.append(resp.json())
            except requests.exceptions.HTTPError as e2:
                print(f"Brave API error (offset {offset}): {e2}")
                if e2.response is not None and e2.response.status_code == 402:
                    raise BravePaymentError("Brave API returned 402 Payment Required.")
                continue
            except Exception as e2:
                print(f"Brave API error (offset {offset}): {e2}")
                continue

    # Merge web results from all offsets
    merged_web_results = []
    for result_json in all_offset_results:
        if 'web' in result_json and 'results' in result_json['web']:
            for search_ele in result_json['web']['results']:
                merged_web_results.append({
                    'title':       search_ele.get('title'),
                    'description': search_ele.get('description'),
                    'url':         search_ele.get('url'),
                })

    if not merged_web_results:
        print("Brave search gave back 0 results.")
    return merged_web_results


def brave_search_batch(queries, freshness="2025-02-01", num_pages=1, max_workers=16):
    """
    Run brave_search for multiple queries in parallel using threads.

    Args:
        queries: dict {key: query_string} or list of query strings.
            If a dict, returns {key: results_list}.
            If a list, returns {query: results_list}.
        freshness: cutoff date passed to brave_search.
        num_pages: number of pages per query.
        max_workers: max threads for the ThreadPoolExecutor.

    Returns:
        dict mapping key (or query string) -> list of search result dicts.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if isinstance(queries, dict):
        key_query_pairs = list(queries.items())
    else:
        key_query_pairs = [(q, q) for q in queries]

    results = {}

    def _search(key, query):
        return key, brave_search(query, freshness=freshness, num_pages=num_pages)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_search, key, query): key
            for key, query in key_query_pairs
        }
        for future in as_completed(futures):
            try:
                key, search_results = future.result()
                results[key] = search_results
            except BravePaymentError:
                raise
            except Exception as e:
                key = futures[future]
                print(f"Brave batch search error for key={key}: {e}")
                results[key] = []

    return results


if __name__ == "__main__":
    import pickle
    import sys
    from pathlib import Path

    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python brave_search_single_query.py "
            '"<query>" <output_file_path.pkl>'
        )

    query = sys.argv[1]
    output_file_path = Path(sys.argv[2])

    results = brave_search(query)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("wb") as f:
        pickle.dump(results, f)

    print(f"Saved {len(results)} results to {output_file_path}")
