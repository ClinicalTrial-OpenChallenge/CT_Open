from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import nullcontext
import os
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse
import pickle
from parse_html_page_utils import _walk_with_paths, \
    filter_researchgate_html, remove_similar_clinical_trials_ichgcp, \
        remove_similar_clinical_trials_non_ichgcp, remove_pubmed_tail, \
        remove_footer, remove_newsletter, filter_marketing_slogans, \
        filter_withpower_junk, filter_academia_e, filter_reference_text, normalize_to_url_html_map\

from typing import Dict, Any, List, Tuple
import traceback

# ---------------------------
# Worker: process a single URL
# ---------------------------
def _process_single_url(url_html_pair: Tuple[str, str]) -> Tuple[str, Dict[str, Any], List[Tuple[str, str]], bool]:
    """
    Process one URL's HTML into:
      - my_dict: {'ref': [count, indices], 'bib': [...], 'coi': [...] } or None if fully handled earlier
      - filtered_results_list: list[(text, path)]
      - is_partial_processed: True for domains we finalize immediately (withpower, academia.e), False otherwise
    """
    url, curr_raw_html = url_html_pair

    # Skip classic.cl entirely (as in your original code)
    if 'classic.cl' in url.lower():
        return url, None, [], True  # treat as partial/ignored

    soup = BeautifulSoup(curr_raw_html, 'html.parser')
    for t in soup(["script", "style", "noscript"]):
        t.decompose()

    results_list: List[Tuple[str, str]] = []
    _walk_with_paths(soup.body or soup, results_list)

    # researchgate trimming
    if 'researchgate' in url.lower():
        filtered_results_list = filter_researchgate_html(results_list)
    else:
        filtered_results_list = results_list

    # strip nav/aside/etc.
    filtered_results_list = [ele for ele in filtered_results_list if 'nav >' not in ele[1]]
    filtered_results_list = [ele for ele in filtered_results_list if 'aside >' not in ele[1]]
    filtered_results_list = [ele for ele in filtered_results_list if 'bsp-header >' not in ele[1]]
    filtered_results_list = [ele for ele in filtered_results_list if 'bsp-list-loadmore >' not in ele[1]]

    # similar trials
    if 'ichgcp.net' in url.lower():
        filtered_results_list = remove_similar_clinical_trials_ichgcp(filtered_results_list)
    else:
        filtered_results_list = remove_similar_clinical_trials_non_ichgcp(filtered_results_list)

    filtered_results_list = remove_pubmed_tail(filtered_results_list)
    filtered_results_list = remove_footer(filtered_results_list)
    filtered_results_list = remove_newsletter(filtered_results_list)
    filtered_results_list = filter_marketing_slogans(filtered_results_list)

    # Fully finalize for withpower / academia.e (no reference dissection later)
    if 'withpower' in url.lower():
        filtered_results_list = filter_withpower_junk(filtered_results_list)
        return url, None, filtered_results_list, True

    if 'academia.e' in url.lower():
        filtered_results_list = filter_academia_e(filtered_results_list)
        return url, None, filtered_results_list, True

    # Otherwise do reference-section detection; keep for LLM disambiguation later
    filtered_results_list, my_dict = filter_reference_text(filtered_results_list)
    return url, my_dict, filtered_results_list, False


# --------------------------------------------
# Parallel driver: parse html pages (pre-LLM)
# --------------------------------------------
def parse_html_page_parallel(
    url2raw: Dict[str, str],
    response_file_name: str = './pipeline_data/longfilatura_multi_proc.pickle',
    max_workers: int = 16
):
    """
    Parallelizes the 'before generation' stage.
    Returns:
        prompt_dict: {(url, key, idx): prompt_str, ...}
        url2my_dict: {url: (my_dict, filtered_results_list)}
        url2_partial_processed: {url: filtered_results_list}  # domains finalized early
    """
    url2my_dict: Dict[str, Tuple[dict, List[Tuple[str, str]]]] = {}
    url2_partial_processed: Dict[str, List[Tuple[str, str]]] = {}

    items = [(u, html) for u, html in url2raw.items()]
    errors = []

    # import traceback
    # for fut in as_completed(futures):
    #     batch_len = futures[fut]
    #     try:
    #         p_partial, my_partial, early_partial, processed = fut.result()
    #     except Exception as e:
    #         print(f"[batch failed] size={batch_len} error={e!r}", flush=True)
    #         traceback.print_exc()
    #         if pbar and hasattr(pbar, "update"):
    #             pbar.update(batch_len)
    #         continue

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_process_single_url, pair): pair[0] for pair in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing html pages (parallel)"):
            url = futures[fut]
            try:
                url, my_dict, filtered_results_list, is_partial = fut.result()
                # Skip the classic.cl no-op worker result
                if filtered_results_list == [] and is_partial and my_dict is None and 'classic.cl' in url.lower():
                    continue

                if is_partial:
                    url2_partial_processed[url] = filtered_results_list
                else:
                    url2my_dict[url] = (my_dict, filtered_results_list)
            except Exception as e:
                errors.append((url, repr(e)))
        # #     import traceback
        # for fut in tqdm(as_completed(futures), total=len(futures), desc="Parsing html pages (parallel)"):
        #     batch_len = futures[fut]
        #     try:
        #         p_partial, my_partial, early_partial, processed = fut.result()
        #     except Exception as e:
        #         print(f"[batch failed] size={batch_len} error={e!r}", flush=True)
        #         traceback.print_exc()
        #         # if pbar and hasattr(pbar, "update"):
        #         #     pbar.update(batch_len)
        #         continue

    # Build prompts for URLs that still need the LLM reference-section disambiguation
    prompt_ref = """
    Determine if the given text is the very beginning of the {key} section of a biomedical research paper based on your knowledge of how the very beginning of the {key} section should look like. Sometimes the latter part of the given text could be from the next section or unrelated parts of a webpage, that should not be take into account.

    Text:
    {content}

    Think step by step. Output your reasoning after the keyword "REASON:"
    Based on your reasoning, output either "yes" or "no" or "unsure" after the keyword "OUTPUT:"
    """

    key2full_name = {'ref': 'Reference', 'bib': 'Bibliography', 'coi': 'Conflict of Interest'}
    prompt_dict: Dict[Tuple[str, str, int], str] = {}

    for url, (my_dict, filtered_results_list) in tqdm(url2my_dict.items(), desc="Constructing prompt dict"):
        if my_dict is None:
            continue
        for key, (count, indices) in my_dict.items():
            if count > 1:
                for idx in indices:
                    # Take ~100 tokens worth of text (approx by words)
                    content_slice = filtered_results_list[idx: min(len(filtered_results_list), idx + 101)]
                    content = ' '.join(ele[0] for ele in content_slice)
                    words = content.split(' ')[:100]
                    content = ' '.join(words)
                    prompt_dict[(url, key, idx)] = prompt_ref.format(content=content, key=key2full_name[key])

    return prompt_dict, url2my_dict, url2_partial_processed


# -----------------
# __main__ updated
# -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True, help="Path to pickle with {'scraped_results_batch': {...}}")
    parser.add_argument('--max_workers', type=int, default=16)
    args = parser.parse_args()

    with open(args.input_file, 'rb') as f:
        raw_url2raw = pickle.load(f)

    assert "scraped_results_batch" in raw_url2raw
    url2raw = normalize_to_url_html_map(raw_url2raw)

    prompt_dict, url2my_dict, url2_partial_processed = parse_html_page_parallel(
        raw_url2raw,
        max_workers=args.max_workers
    )

    # with open('./pipeline_data/longfilatura_multi_process_test.pickle', 'wb') as f:
    #     pickle.dump({'prompt_dict': prompt_dict, 'url2my_dict': url2my_dict, 'url2_partial_processed': url2_partial_processed}, f)



    # --- generation happens later ---
    # res, time_out, _ = generate(..., prompt_dict=prompt_dict, ...)
    # (rest of your existing generation logic)
