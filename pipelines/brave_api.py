# multithreaded_brave_search.py  (robust version)
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import pickle
import concurrent.futures
import argparse
import os

API_TOKEN = os.getenv("BRAVE_API_TOKEN")
if API_TOKEN is None:
    raise ValueError("BRAVE_API_TOKEN is not set")
MAX_WORKERS = 5


import json
from datetime import datetime
from dateutil.parser import parse

def parse_google_search_results(search_result_json):
    """
    Parses a JSON response from the Google Custom Search API to extract
    titles, links, snippets, and the earliest publication date for each result.

    It checks multiple metadata fields for dates, parses them, and selects
    the earliest one found. Requires the 'python-dateutil' library.

    Args:
        search_result_json (dict): A Python dictionary representing the
                                   JSON response from the API.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              'title', 'url', 'description', and 'date' of a search result.
              The 'date' is the earliest found date in 'YYYY-MM-DD' format,
              or None if no valid date is available. Returns an empty list
              if no items are found.
    """
    parsed_results = []
    # Check if the 'items' key exists and is a list in the JSON response.
    if 'items' not in search_result_json or not isinstance(search_result_json['items'], list):
        return parsed_results

    # Iterate over each item in the results list
    for item in search_result_json['items']:
        result = {
            'title': item.get('title', 'N/A'),
            'url': item.get('link', 'N/A'),
            'description': item.get('snippet', 'N/A')
        }

        # --- Date extraction logic ---
        earliest_date = None
        found_dates = []
        pagemap = item.get('pagemap', {})

        if 'metatags' in pagemap and isinstance(pagemap['metatags'], list) and len(pagemap['metatags']) > 0:
            metatags = pagemap['metatags'][0]
            # List of common keys for publication date
            date_keys = [
                'citation_publication_date',
                'article:published_time',
                'publication_date',
                'date',
                'creationdate',
                'citation_online_date',
                'dc.date'
            ]

            for key in date_keys:
                if key in metatags:
                    date_string = metatags[key]
                    # Ensure the date_string is a non-empty string before parsing
                    if date_string and isinstance(date_string, str):
                        # Handle specific non-standard formats like "D:20240127..."
                        if date_string.startswith('D:'):
                            date_string = date_string.lstrip('D:')
                        try:
                            # Use the powerful parser from dateutil to handle various formats
                            # The 'ignoretz=True' parameter simplifies comparison
                            parsed_dt = parse(date_string, ignoretz=True)
                            found_dates.append(parsed_dt)
                        except (ValueError, TypeError, OverflowError):
                            # Ignore dates that can't be parsed
                            continue
        
        # If any dates were successfully found and parsed, find the minimum (earliest)
        if found_dates:
            earliest_dt_object = min(found_dates)
            # This line formats the date into the required 'YYYY-MM-DD' string
            earliest_date = earliest_dt_object.strftime('%Y-%m-%d')

        result['date'] = earliest_date
        parsed_results.append(result)

    return parsed_results

def get_brave_results(my_query,earliest_date, study_start_date, offset):
    params = {
                "q":              my_query,
                "country":        "ALL",
                "safesearch":     "off",
                "spellcheck":     "false",
                "extra_snippets": "true",
                "offset":         offset,
                "search_lang":    "en",
                "ui_lang":        "en-US",
            }
    if FRESHNESS_FLAG == "True":
        params['freshness'] = f"{study_start_date}to{earliest_date}"
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
    return resp
    
    

def fetch_single(nctid,my_query,earliest_date,study_start_date):
    """
    Try to fetch Brave results for one (nctid, ele) pair across offsets 0-4.
    Returns (success_flag, nctid, payload_or_errmsg)
    """

    all_offset_results = []
    fell_back_to_google = False

    for offset in range(5):  # offsets 0-4
        try:
            resp = get_brave_results(my_query, earliest_date, study_start_date, offset)
            resp.raise_for_status()
            all_offset_results.append(resp.json())
        except Exception as e:
            try:
                resp = get_brave_results(my_query, earliest_date, study_start_date, offset)
                resp.raise_for_status()
                all_offset_results.append(resp.json())
            except Exception as e:
                return (False, nctid, f"All attempts failed: {e}")

    if not all_offset_results:
        return (False, nctid, "No results from any offset or fallback")

    # Merge Brave results across offsets: concatenate web results
    merged = all_offset_results[0]
    for subsequent in all_offset_results[1:]:
        if 'web' in subsequent and 'results' in subsequent['web']:
            if 'web' not in merged:
                merged['web'] = {'results': []}
            elif 'results' not in merged['web']:
                merged['web']['results'] = []
            merged['web']['results'].extend(subsequent['web']['results'])

    return (True, nctid, ('brave', {
        "query":            my_query,
        "earliest_date":    earliest_date,
        "study_start_date": study_start_date,
        "results":          merged,
    }))



# Assume MAX_WORKERS is defined, e.g., MAX_WORKERS = 10
# Assume 'query_type' is a global variable (1, 2, or 3)
# Assume the 'fetch_single' function is defined as in your example

def threaded_brave_search(random_data):
    """
    Runs Brave searches in parallel using a structured data input.

    This function takes a dictionary where each item contains all necessary
    information for a search, including the NCTID, the full data element ('ele'),
    multiple query types, and search metadata.

    Args:
        random_data (dict): A dictionary mapping NCTIDs to their data.
            Example structure:
            {
                "NCT000001": {
                    "ele": { ... },
                    "earliest_date": "2024-01-15" or None,
                    "brief_title": "Brief title query",
                    "nct_id": "nctid as query",
                    "llm_rewrite": "llm rewritten query",
                    "search_round": 0
                }
            }

    Returns:
        tuple: A tuple containing:
            - nctid2results (dict): A dictionary of successful search results.
            - failed_nctids (list): A list of (nctid, error_message) tuples for failed searches.
    """
    nctid2results = {}
    failed_nctids = []
    tasks_to_submit = []

    # 1. Prepare all tasks before submitting to the thread pool
    for nctid, data in random_data.items():
        # --- Select the query based on the global variable ---
        my_query = data[query_type]

        # --- Determine the earliest date for the search ---
        earliest_date = data.get("earliest_date")
        study_start_date = data['ele'].get('protocolSection', {}).get('statusModule', {}).get('startDateStruct', {}).get('date', '')

        # If earliest_date is None, calculate it using the provided logic
        if earliest_date is None and data['search_round'] == 0:
            # print(f"running earliest date none round 0 for {nctid}")
            ele = data['ele']
            status_mod = ele['protocolSection']['statusModule']
            keys = ["completionDateStruct", "primaryCompletionDateStruct", "resultsFirstSubmitDate"]
            
            # Safely extract all available dates from the 'ele' object
            dates = [
                (status_mod[k] if k == "resultsFirstSubmitDate" else status_mod[k].get('date'))
                for k in keys if k in status_mod and status_mod.get(k)
            ]
            # Filter out any None values that may have resulted from .get()
            valid_dates = [d for d in dates if d]

            if valid_dates:
                # Per instructions, use the minimum of the study dates and a fixed date.
                # Note: min(min(dates), "YYYY-MM-DD") is equivalent to min(dates + ["YYYY-MM-DD"])
                earliest_date = min(valid_dates + [cutoff_date])
            else:
                # Default date if no dates are found in the trial data
                earliest_date = cutoff_date
        elif earliest_date is None and data['search_round'] > 0:
            # print(f"running earliest date none round > 0 for {nctid}")
            # TODO: Potentially use these points
            no_date_nctids.append(nctid)
            continue
        # else:
        #     print(f"running date given for {nctid}")

        
        tasks_to_submit.append((nctid, my_query, earliest_date, study_start_date))

    # 2. Execute the prepared tasks in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        # Create a map of futures to their original NCTID
        futures = {
            pool.submit(fetch_single, nctid, query, date, study_start_date): nctid
            for nctid, query, date, study_start_date in tasks_to_submit
        }

        progress_bar = tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="BraveAPI search")
        
        for fut in progress_bar:
            try:
                success, result_nctid, payload = fut.result()
                if success:
                    nctid2results[result_nctid] = payload
                else:
                    # The payload is the error message on failure
                    failed_nctids.append((result_nctid, payload))
            except Exception as e:
                # Handle exceptions raised by the future itself
                failed_nctid = futures[fut]
                failed_nctids.append((failed_nctid, f"Thread execution failed: {e}"))

    return nctid2results, failed_nctids


# ------------------------- Example usage -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--query_type", type=str, required=True, choices=["brief_title", "nct_id", "llm_rewrite","sponsor_acronym"])
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--FRESHNESS_FLAG", type=str, required=True)
    parser.add_argument("--cutoff_date", type=str, required=True, help="Cutoff date in YYYY-MM-DD format, e.g. 2026-03-01")
    args = parser.parse_args()
    query_type = args.query_type
    exp_name = args.exp_name
    print("offset: 0-4 (all fetched)")
    print("FRESHNESS_FLAG: ", args.FRESHNESS_FLAG)
    FRESHNESS_FLAG = args.FRESHNESS_FLAG
    cutoff_date = args.cutoff_date
    no_date_nctids = []
    
    # with open('./jul19_ct_likely_test_set_data_34727.pickle', 'rb') as f:
    #     data = pickle.load(f)

    with open(args.input_path, 'rb') as f:
        data = pickle.load(f)

    nctid2results, failed = threaded_brave_search(data)

    print(f"Completed: {len(nctid2results)}")
    print(f"Failed:    {len(failed)}")
    
    # Save the results and failed cases to a single pickle file
    print("Start saving...")
    data_s1 =  {
                "nctid2results": nctid2results,
                "failed_data": {ele[0]: data[ele[0]] for ele in failed},
                "failed_payload": [ele[1] for ele in failed],
            }
    # old path './jul19_ct_likely_test_set_data_34727_brave_search_results.pickle'
    with open(f'{exp_name}_{query_type}_step_1.pickle', 'wb') as f:
        pickle.dump(data_s1, f)
    
    with open(f'{exp_name}_{query_type}_step_1_no_date_nctids.pickle','wb') as f:
        pickle.dump(no_date_nctids, f)

    data_s1_5 = {}
    for k in data:
        if k in no_date_nctids:
            continue
        data_s1_5[k] = []
        if data_s1['nctid2results'][k][0] == 'brave':
            results = data_s1['nctid2results'][k][1]['results']
            if 'web' in results and 'results' in results['web']:
                for search_ele in results['web']['results']:
                    temp = {}
                    temp['title'] = search_ele['title'] if 'title' in search_ele else None
                    temp['description'] = search_ele['description'] if 'description' in search_ele else None
                    temp['url'] = search_ele['url'] if 'url' in search_ele else None
                    if 'age' in search_ele:
                        temp['date'] = search_ele['age']
                    else:
                        temp['date'] = None
                    data_s1_5[k].append(temp)
        else:
            assert data_s1['nctid2results'][k][0] == 'google'
            data_s1_5[k] = parse_google_search_results(data_s1['nctid2results'][k][1])

    with open(f'{exp_name}_{query_type}_step_1_5.pickle', 'wb') as f:
        pickle.dump(data_s1_5, f)
    print("Done!")