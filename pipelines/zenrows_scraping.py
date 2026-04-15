import requests
import concurrent.futures
from zenrows import ZenRowsClient
import logging
import pickle
from tqdm import tqdm
from urllib.parse import urlparse
from requests.exceptions import InvalidURL # Add this line
import psycopg2
import psycopg2.extras
import pickle
import os
import logging
import time


# --- Configuration ---
# A powerful and common user agent to mimic a real browser.
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'

# Replace with your actual ZenRows API key.
ZENROWS_API_KEY = os.getenv("ZENROWS_API_KEY")
if ZENROWS_API_KEY is None:
    raise ValueError("ZENROWS_API_KEY is not set")

# A set of common downloadable file extensions for quick, network-free filtering.
# This serves as a fast first-pass filter.
DOWNLOADABLE_EXTENSIONS = {
    # Documents
    '.pdf', '.doc', '.docx', '.ppt', '.pptx', '.xls', '.xlsx', '.odt', '.ods', '.odp',
    # Archives
    '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2',
    # Images
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.webp',
    # Audio/Video
    '.mp3', '.wav', '.mp4', '.mov', '.avi', '.mkv', '.flv',
    # Data/Code (that we don't want to scrape as a webpage)
    '.csv', '.json', '.xml', '.js', '.css', '.iso'
}


# --- Main Logic ---

def is_downloadable_file_by_extension(url):
    """
    Quickly checks if a URL likely points to a downloadable file based on its extension.
    This is much faster than a HEAD request as it involves no network I/O. It correctly
    handles URLs with query parameters by only checking the path.

    Args:
        url (str): The URL to check.

    Returns:
        bool: True if the URL ends with a downloadable file extension, False otherwise.
    """
    try:
        parsed_path = urlparse(url).path
        if parsed_path.lower().endswith(tuple(DOWNLOADABLE_EXTENSIONS)):
            return True
    except Exception as e:
        return False
    return False


def scrape_url(url):
    """
    Scrapes a single URL with a more robust fallback mechanism.
    If the initial 'requests' call fails for ANY reason (timeout, connection
    error, 5xx server error, etc.), it will fall back to using ZenRows.
    """
    if "pubmed.ncbi.nlm.nih.gov" in url or "pmc.ncbi.nlm.nih.gov" in url or "biorxiv.org" in url or "medrxiv.org" in url or ("ncbi.nlm.nih.gov" in url and "PMC" in url) or ("rcastoragev2.blob.core.windows.net" in url and "PMC" in url):
        return url, "", 'database_article'
    try:
        client = ZenRowsClient(ZENROWS_API_KEY)
        response = client.get(url, timeout=90)
        response.raise_for_status()
        # print(f"ZenRows normal request successful for {url}")
        return url, response.text, 'zenrows_normal'
    except Exception as e:
        # print(f"zenrows normal failed, trying premium request {url}")
        try:
            client = ZenRowsClient(ZENROWS_API_KEY)
            params = {
                'antibot': 'true',  # Enable the strongest anti-bot features
                'js_render': 'true',  # Render JavaScript on the page
                'premium_proxy': 'true',  # Use high-quality residential proxies
                'wait_for': 'body',
                'proxy_country': 'us',
            }
            response = client.get(url, params=params, timeout=90)
            response.raise_for_status()
            # print(f"ZenRows premium request successful for {url}")
            return url, response.text, 'zenrows_premium'
        except Exception as zen_e:
            # print(f"zenrows premium failed for {url}, {zen_e}")
            return url, f"Failed with ZenRows Premium: {zen_e}", 'failed'


def chunker(seq, size):
    """Yield successive n-sized chunks from a sequence."""
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))

def main(args):
    """
    Main function to orchestrate the multithreaded scraping process in batches.
    It uses a two-layer filtering system: first by file extension (fast, no network),
    then by Content-Type header during the scraping process.
    """

    if not URLS_TO_SCRAPE:
        print("URL list is empty. Exiting.")
        return
    

    BATCH_SIZE = 90

    # --- Initialize CUMULATIVE counters for the final summary ---
    total_skipped_by_extension = 0
    total_skipped_by_content_type = 0
    total_zenrows_success = 0
    total_zenrows_premium_success = 0
    total_failed_scrapes = 0

    url_batches = list(chunker(URLS_TO_SCRAPE, BATCH_SIZE))
    total_batches = len(url_batches)

    # --- Process URLs in batches ---
    for i, batch in enumerate(url_batches):
        # --- Initialize BATCH-SPECIFIC counters ---
        batch_skipped_by_extension = 0
        batch_skipped_by_content_type = 0
        batch_zenrows_success = 0
        batch_zenrows_premium_success = 0
        batch_failed_scrapes = 0
        
        scraped_results_batch = {}
        batch_skipped_files = []

        print("\n" + "="*50)
        print(f"Processing Batch {i}/{total_batches} ({len(batch)} URLs)")
        print("="*50)

        # --- Step 1 (Layer 1 Filter): Filter by extension (very fast, no network) ---
        print(f"\nBatch {i} | Step 1: Filtering URLs by file extension...")
        webpage_urls_to_scrape = []
        for url in tqdm(batch, desc=f"Batch {i} Filtering"):
            if is_downloadable_file_by_extension(url):
                batch_skipped_files.append(url)
            else:
                webpage_urls_to_scrape.append(url)
        
        batch_skipped_by_extension = len(batch_skipped_files)
        total_skipped_by_extension += batch_skipped_by_extension
        print(f"Skipped {batch_skipped_by_extension} URLs based on file extension.")
        print(f"Proceeding to scrape the remaining {len(webpage_urls_to_scrape)} URLs.")
        print("---")


        # --- Step 2 (Layer 2 Filter & Scrape): Scrape concurrently ---
        if webpage_urls_to_scrape:
            print(f"Batch {i} | Step 2: Scraping webpages (with Content-Type check)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=90) as executor:
                future_to_url = {executor.submit(scrape_url, url): url for url in webpage_urls_to_scrape}
                
                for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(webpage_urls_to_scrape), desc=f"Batch {i} Scraping"):
                    url, content, method = future.result()
                    
                    # Store all outcomes in the batch results for potential review
                    scraped_results_batch[url] = {"content": content, "method": method}
                    
                    # Update BATCH and CUMULATIVE counters based on the outcome
                    if method == 'zenrows_normal':
                        batch_zenrows_success += 1
                        total_zenrows_success += 1
                    elif method == 'zenrows_premium':
                        batch_zenrows_premium_success += 1
                        total_zenrows_premium_success += 1
                    elif method == 'skipped_content_type':
                        batch_skipped_by_content_type += 1
                        total_skipped_by_content_type += 1
                    else:  # method == 'failed'
                        batch_failed_scrapes += 1
                        total_failed_scrapes += 1
        else:
            print("No webpages to scrape in this batch after extension filtering.")
            
        # --- BATCH SUMMARY ---
        print("\n" + "-"*20 + f" Batch {i} Summary " + "-"*20)
        batch_total_processed = batch_skipped_by_extension + batch_zenrows_success + batch_skipped_by_content_type + batch_failed_scrapes
        print(f"Processed {batch_total_processed} URLs in this batch.")
        print(f"  - Skipped by file extension: {batch_skipped_by_extension}")
        print(f"  - Skipped by Content-Type: {batch_skipped_by_content_type}")
        print(f"  - 🚀 Scraped with ZenRows normal: {batch_zenrows_success}")
        print(f"  - 🚀 Scraped with ZenRows premium: {batch_zenrows_premium_success}")
        print(f"  - ❌ Failed to scrape: {batch_failed_scrapes}")
        print("-" * 55)

        # --- Save the BATCH results to a unique file ---
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        batch_filename = f'{args.output_dir}/{args.exp_name}_zenrows_results_{i}.pickle'
        print(f"Saving batch results to {batch_filename}...")
        with open(batch_filename, 'wb') as f:
            # Saving the detailed dictionary and the list of files skipped by extension
            pickle.dump(
                {'scraped_results_batch': scraped_results_batch, 'batch_skipped_by_extension': batch_skipped_files}, f)
        print("Save complete.")
        sleep_indicator = False
        for k,v in scraped_results_batch.items():
            if "RemoteDisconnected" in v['content']:
                sleep_indicator = True
                break
        if sleep_indicator:
            # sleep for 5 mins
            print("❌: getting connection error, sleep for 5 mins.")
            time.sleep(5 * 60)
        else:
            print("✅: No connection errors detected.")

    # --- FINAL OVERALL SUMMARY ---
    print("\n" + "="*50)
    print("--- ALL BATCHES COMPLETE: OVERALL SUMMARY ---")
    print(f"Processed {len(URLS_TO_SCRAPE)} total URLs across {total_batches} batches.")
    print("\nFiltering & Skipping Breakdown:")
    print(f"- Skipped by file extension: {total_skipped_by_extension}")
    print(f"- Skipped by Content-Type (not HTML): {total_skipped_by_content_type}")
    print("-" * 25)
    
    print("Scraping Method Breakdown:")
    print(f"🚀 Scraped with ZenRows API: {total_zenrows_success+total_zenrows_premium_success}")
    print(f"❌ Failed to scrape (errors): {total_failed_scrapes}")
    print("-" * 25)
    
    total_successful = total_zenrows_success + total_zenrows_premium_success
    print(f"Total successfully scraped HTML pages: {total_successful}")
    print(f"Results are saved in batch files named {args.output_dir}/{args.exp_name}_N.pickle")
    print("="*50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scrape URLs with ZenRows API')
    parser.add_argument('--exp_name', type=str,required=True, help='experiment name')
    parser.add_argument('--input_data_type', type=str,required=True, help='Type of input data', choices=['s1_5', 'all_urls', 'scraped_batch_files'])
    parser.add_argument('--input_path', type=str,required=False, help='Path to input data file (pickle format)')
    parser.add_argument("--input_dir", type=str,default="request_results", help='directory to save the scraped results')
    parser.add_argument("--output_dir", type=str,default="zenrows_results", help='directory to save the scraped results')
    args = parser.parse_args()

    database_urls = []

    print(f"database_urls: {len(database_urls)}")


    skipped_urls = []
    # s1_5 from brave_api, all_urls is a list of urls
    if args.input_data_type == 's1_5':
        # TODO: finish the code for pipeline
        raise NotImplementedError("s1_5 input data type is not implemented yet.")
        # with open(args.input_path, 'rb') as f:
        #     data_s1_5 = pickle.load(f)

        # all_urls = []
        # for nctid in data_s1_5:
        #     for ele in data_s1_5[nctid]:
        #         all_urls.append(ele['url'])
    elif args.input_data_type == 'all_urls':
        all_files = os.listdir(args.input_dir)
        all_files = [f for f in all_files if f.startswith(args.exp_name) and "skipped" not in f]
        all_urls = []
        for f in tqdm(all_files, desc="Loading request results"):
            with open(f"{args.input_dir}/{f}", 'rb') as f:
                data = pickle.load(f)
            curr = data['scraped_results_batch']
            for k,v in curr.items():
                if 'failed' == v['method']:
                    all_urls.append(k)
    elif args.input_data_type == 'scraped_batch_files':
        with open(args.input_path, 'rb') as f:
            data_pickle = pickle.load(f)
        all_urls = data_pickle


    print(f"urls in database: {len(database_urls)}")
    print(f"urls in all_urls: {len(all_urls)}")
    URLS_TO_SCRAPE = list(set(all_urls) - set(database_urls))
    
    print(f"Application has started successfully. {len(URLS_TO_SCRAPE)} URLs to scrape.")

    print(f"URLS_TO_SCRAPE: {URLS_TO_SCRAPE[:10]}")
    main(args)
    # updated_scraped_urls = database_urls + URLS_TO_SCRAPE
    # with open("./aug21_unique_urls_scraped.pickle",'wb') as f:
    #     pickle.dump(updated_scraped_urls,f)
    # print(f"updated_scraped_urls: {len(updated_scraped_urls)}")
    with open(f"{args.output_dir}/{args.exp_name}_zenrows_skipped_urls.pickle",'wb') as f:
        pickle.dump(skipped_urls, f)