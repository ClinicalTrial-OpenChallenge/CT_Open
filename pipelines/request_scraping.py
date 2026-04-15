import requests
import concurrent.futures
import logging
import pickle
from tqdm import tqdm
from urllib.parse import urlparse
from requests.exceptions import InvalidURL # Add this line
import psycopg2
import psycopg2.extras
import pickle
import logging


# --- Configuration ---
# A powerful and common user agent to mimic a real browser.
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/115.0'

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
    try:
        # Initial attempt with standard requests
        headers = {'User-Agent': USER_AGENT}
        response = requests.get(url, headers=headers, timeout=20, stream=True)
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '').lower()
        if 'text/html' not in content_type:
            skipped_urls.append(url)
            return url, "", 'skipped_content_type'
        # print(f"requests successful for {url}")
        return url, response.text, 'requests'
    except Exception as e:
        # print(f"Failed: An UNEXPECTED error occurred of type {type(e).__name__}: {e}")
        return url, f"Failed: An UNEXPECTED error occurred of type {type(e).__name__}: {e}", 'failed'


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
    

    BATCH_SIZE = 10000

    # --- Initialize CUMULATIVE counters for the final summary ---
    total_skipped_by_extension = 0
    total_skipped_by_content_type = 0
    total_requests_success = 0
    total_failed_scrapes = 0

    url_batches = list(chunker(URLS_TO_SCRAPE, BATCH_SIZE))
    total_batches = len(url_batches)

    # --- Process URLs in batches ---
    for i, batch in enumerate(url_batches):
        # --- Initialize BATCH-SPECIFIC counters ---
        batch_skipped_by_extension = 0
        batch_skipped_by_content_type = 0
        batch_requests_success = 0
        batch_failed_scrapes = 0
        scraped_results_batch = {}
        batch_skipped_files = []

        print("\n" + "="*50)
        print(f"Processing Batch {i + 1}/{total_batches} ({len(batch)} URLs)")
        print("="*50)

        # --- Step 1 (Layer 1 Filter): Filter by extension (very fast, no network) ---
        print(f"\nBatch {i+1} | Step 1: Filtering URLs by file extension...")
        webpage_urls_to_scrape = []
        for url in tqdm(batch, desc=f"Batch {i+1} Filtering"):
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
            print(f"Batch {i+1} | Step 2: Scraping webpages (with Content-Type check)...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=90) as executor:
                future_to_url = {executor.submit(scrape_url, url): url for url in webpage_urls_to_scrape}
                
                for future in tqdm(concurrent.futures.as_completed(future_to_url), total=len(webpage_urls_to_scrape), desc=f"Batch {i+1} Scraping"):
                    url, content, method = future.result()
                    
                    # Store all outcomes in the batch results for potential review
                    scraped_results_batch[url] = {"content": content, "method": method}
                    
                    # Update BATCH and CUMULATIVE counters based on the outcome
                    if method == 'requests':
                        batch_requests_success += 1
                        total_requests_success += 1
                    elif method == 'skipped_content_type':
                        batch_skipped_by_content_type += 1
                        total_skipped_by_content_type += 1
                    else:  # method == 'failed'
                        batch_failed_scrapes += 1
                        total_failed_scrapes += 1
        else:
            print("No webpages to scrape in this batch after extension filtering.")
            
        # --- BATCH SUMMARY ---
        print("\n" + "-"*20 + f" Batch {i + 1} Summary " + "-"*20)
        batch_total_processed = batch_skipped_by_extension + batch_requests_success + batch_skipped_by_content_type + batch_failed_scrapes
        print(f"Processed {batch_total_processed} URLs in this batch.")
        print(f"  - Skipped by file extension: {batch_skipped_by_extension}")
        print(f"  - Skipped by Content-Type: {batch_skipped_by_content_type}")
        print(f"  - ✅ Scraped with 'requests': {batch_requests_success}")
        print(f"  - ❌ Failed to scrape: {batch_failed_scrapes}")
        print("-" * 55)

        # --- Save the BATCH results to a unique file ---

        import os
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        batch_filename = f'{args.output_dir}/{args.exp_name}_request_results_{i + 1}.pickle'
        print(f"Saving batch results to {batch_filename}...")
        with open(batch_filename, 'wb') as f:
            # Saving the detailed dictionary and the list of files skipped by extension
            pickle.dump(
                {'scraped_results_batch': scraped_results_batch, 'batch_skipped_by_extension': batch_skipped_files}, f)
        print("Save complete.")

    # --- FINAL OVERALL SUMMARY ---
    print("\n" + "="*50)
    print("--- ALL BATCHES COMPLETE: OVERALL SUMMARY ---")
    print(f"Processed {len(URLS_TO_SCRAPE)} total URLs across {total_batches} batches.")
    print("\nFiltering & Skipping Breakdown:")
    print(f"- Skipped by file extension: {total_skipped_by_extension}")
    print(f"- Skipped by Content-Type (not HTML): {total_skipped_by_content_type}")
    print("-" * 25)
    
    print("Scraping Method Breakdown:")
    print(f"✅ Scraped with standard 'requests': {total_requests_success}")
    print(f"❌ Failed to scrape (errors): {total_failed_scrapes}")
    print("-" * 25)
    
    total_successful = total_requests_success
    print(f"Total successfully scraped HTML pages: {total_successful}")
    print(f"Results are saved in batch files named {args.output_dir}/{args.exp_name}_request_results_N.pickle")
    print("="*50)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scrape URLs with ZenRows API')
    parser.add_argument('--exp_name', type=str,required=True, help='Path to save the scraped results')
    parser.add_argument('--input_path', type=str,required=True, help='input_path')
    parser.add_argument('--input_data_type', type=str,required=True, help='Type of input data', choices=['s1_5', 'all_urls'])
    parser.add_argument("--todays_date", type=str,required=True, help='file name of today running url path e.g. aug22')
    parser.add_argument("--output_dir", type=str,default="request_results", help='directory to save the scraped results')
    args = parser.parse_args()

    # database_urls = [s[0] for s in database_urls]
    database_urls = []
    # with open("./aug21_unique_urls_scraped.pickle",'rb') as f:
    #     database_urls = pickle.load(f)

    print(f"database_urls: {len(database_urls)}")


    skipped_urls = []
    # s1_5 from brave_api, all_urls is a list of urls
    if args.input_data_type == 's1_5':
        with open(f'data/{args.todays_date}_{args.exp_name}_step_1_5.pickle','rb') as f:
            file = pickle.load(f)
        all_urls = []
        for nctid in file:
            for s_res in file[nctid]:
                all_urls.append(s_res['url'])
    elif args.input_data_type == 'all_urls':
        # with open(f'data/{args.todays_date}_{args.exp_name}_urls.pickle', 'rb') as f:
        with open(f'{args.input_path}', 'rb') as f:
            all_urls = pickle.load(f)
    
    print(f"urls in database: {len(database_urls)}")
    print(f"urls in all_urls: {len(all_urls)}")
    URLS_TO_SCRAPE = list(set(all_urls) - set(database_urls))
    
    print(f"Application has started successfully. {len(URLS_TO_SCRAPE)} URLs to scrape.")

    main(args)
    updated_scraped_urls = database_urls + URLS_TO_SCRAPE
    # with open("./aug21_unique_urls_scraped.pickle",'wb') as f:
    #     pickle.dump(updated_scraped_urls,f)
    print(f"updated_scraped_urls: {len(updated_scraped_urls)}")
    with open(f"{args.output_dir}/{args.exp_name}_request_skipped_urls.pickle",'wb') as f:
        pickle.dump(skipped_urls, f)