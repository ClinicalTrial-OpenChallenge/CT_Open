import string
import copy
import pickle
import random
from typing import List, Tuple
from bs4 import BeautifulSoup, Tag, NavigableString
from tqdm import tqdm
import tiktoken
import numpy as np
encoding = tiktoken.encoding_for_model("gpt-4o")
from generation import generate


def _table_to_markdown(table: Tag) -> List[str]:
    """Convert a <table> element to a GitHub-flavoured Markdown table."""
    rows = table.find_all("tr", recursive=False)
    if not rows:
        return []

    # First row -> header (falling back to <td> if the author skipped <th>)
    header_cells = rows[0].find_all(["th", "td"], recursive=False)
    headers = [c.get_text(" ", strip=True) for c in header_cells]
    out = [
        "|" + "|".join(headers) + "|",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]

    # Remaining rows
    for r in rows[1:]:
        cells = r.find_all(["th", "td"], recursive=False)
        out.append(
            "|" + "|".join(c.get_text(" ", strip=True) for c in cells) + "|"
        )
    return out

def _get_tag_path(tag: Tag) -> str:
    """
    Generates a CSS selector-like path string for a given Beautiful Soup tag.
    Example: 'html > body > div > h1'
    """
    path_parts = []
    # Iterate through the tag's parents up to the root
    for parent in tag.parents:
        if parent.name is None or parent.name == '[document]':
            break

        path_parts.append(parent)

    # The path is from the root down, so reverse the list of parents
    path_parts.reverse()

    # Add the current tag's name to complete the path

    path_parts.append(tag)
    new_path_parts = []
    for p in path_parts:
        if p.name == 'section':
            # Start with the tag name
            identifier = p.name
            # Add the ID if it exists (e.g., #main-content)
            if p.has_attr('id'):
                identifier += f"#{p['id']}"
            # # Add all classes if they exist (e.g., .featured.news)
            # if p.has_attr('class'):
            #     identifier += "." + ".".join(p['class'])
            new_path_parts.append(identifier)
        else:
            new_path_parts.append(p.name)
    # new_path_parts = [p.name if p.name != 'section' else p.name + '+' + p['class'].text for p in path_parts]
    return " > ".join(new_path_parts)

# --- New function that produces a list of tuples ---
def _walk_with_paths(node: Tag, results: List[Tuple[str, str]], indent: int = 0) -> None:
    """
    Depth-first traversal that appends (Markdown_line, tag_path) tuples into results.
    """
    pad = " " * indent

    # --- TEXT (handled by recursive calls from parent tags) ---
    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            # The path for a raw text node is its parent's path
            path = _get_tag_path(node.parent) if node.parent else 'html'
            results.append((text, path))
        return

    if not isinstance(node, Tag):
        return

    name = node.name.lower()

    # For most elements, we can generate the path directly from the node.
    # We do this before the conditional logic to avoid repetition.
    path = _get_tag_path(node)

    # --- ELEMENTS ---
    # Headings
    if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
        level = int(name[1])
        text = "#" * level + " " + node.get_text(" ", strip=True)
        results.append((text, path))

    # Paragraphs
    elif name == "p":
        text = node.get_text(" ", strip=True)
        results.append((text, path))

    # Un-ordered list
    elif name == "ul":
        for li in node.find_all("li", recursive=False):
            # For list items, the specific path is for the <li> tag
            li_path = _get_tag_path(li)
            text = pad + "- " + li.get_text(" ", strip=True)
            results.append((text, li_path))

    # Ordered list
    elif name == "ol":
        for i, li in enumerate(node.find_all("li", recursive=False), start=1):
            # For list items, the specific path is for the <li> tag
            li_path = _get_tag_path(li)
            text = pad + f"{i}. " + li.get_text(" ", strip=True)
            results.append((text, li_path))

    # Table
    # THIS FUNCTION HAS PROBLEM, WE SHOULD NOT USE IT; OR, WE SHOULD FIX IT.
    # elif name == "table":
    #     # Assuming a _table_to_markdown function exists as in the original code.
    #     # Each line of the table is associated with the <table> tag's path.
    #     try:
    #         markdown_lines = _table_to_markdown(node)
    #         for line in markdown_lines:
    #             results.append((line, path))
    #     except NameError:
    #         # Fallback if _table_to_markdown is not defined
    #         results.append(("[Table content would go here]", path))


    # Everything else -> descend recursively
    else:
        for child in node.children:
            # Note: We pass `results` list down, not a buffer.
            _walk_with_paths(child, results, indent)

def filter_researchgate_html(results_list) -> str:
    """
    Filter the html string to exclude similar articles in ResearchGate.

    Args:
        result_list (list): The list of tuples containing the text and path of the filtered html string.
    Returns:
        sliced_result_list (list): The list of tuples containing the text and path of the filtered html string.
        None: If there is no such thing as "Citations (" or "References (" in the html string.
            - In this case, we need to use premium zenrows request to scrape again - if we did not use zenrows premium to scrape.
    """
    # soup = BeautifulSoup(html_str, 'html.parser')
    # for t in soup(["script", "style", "noscript"]):
    #     t.decompose()
    # results_list = []
    # _walk_with_paths(soup.body or soup, results_list)

    citations_idx = [i for i, (curr_text, curr_path) in enumerate(results_list) if 'Citations (' in curr_text]
    references_idx = [i for i, (curr_text, curr_path) in enumerate(results_list) if 'References (' in curr_text]

    if len(citations_idx) == 0 and len(references_idx) == 0:
        return results_list

    if len(citations_idx) == 2:
        sliced_index = citations_idx[1]
    elif len(references_idx) == 2:
        sliced_index = references_idx[1]
    else:
        sliced_index = citations_idx[0]

    sliced_result_list = results_list[:sliced_index]

    return sliced_result_list

def remove_hashtags_colons(text: str) -> str:
    text = text.replace("#", "").replace(':', "")
    return text.strip()

def remove_punctuations(text: str) -> str:
    punctuation_chars = string.punctuation
    no_punct = "".join([char for char in text if char not in punctuation_chars])
    return no_punct.strip()


def filter_reference_text(result_list: list) -> str:
    """
    Filter the reference text from the result list.

    Args:
        result_list (list): The list of tuples containing the text and path of the filtered html string.
    Returns:
        sliced_result_list (list): The list of tuples containing the text and path of the filtered html string.
    """
    sliced_index = None
    count = 0
    my_dict = {
        "ref": [0, []],
        # 'ackn': [0, len(result_list)],
        'bib': [0, []],
        'coi': [0, []],
    }
    reference_indices = []
    for ele_idx, ele in enumerate(result_list):
        new_ele = remove_hashtags_colons(ele[0])
        if new_ele in ['eferences', 'REFERENCES', 'References', 'Referenc', 'IRODAL', 'ЛИТЕР', 'Literatura','Reference(s)']:
            count+=1
            sliced_index = ele_idx
            reference_indices.append(ele_idx)
        elif new_ele == 'Reference' and ele_idx!= len(result_list)-2:
            if result_list[ele_idx+1][0] == 's':
                count+=1
                sliced_index = ele_idx
                reference_indices.append(ele_idx)
        elif new_ele == 'R' and ele_idx!= len(result_list)-10:
            if ''.join([small_ele[0] for small_ele in result_list[ele_idx+1:ele_idx+10]]) == 'eferences':
                count+=1
                sliced_index = ele_idx
                reference_indices.append(ele_idx)
        elif new_ele == 'Re' and ele_idx!= len(result_list)-9:
            if 'ferences' in ''.join([small_ele[0] for small_ele in result_list[ele_idx+1:ele_idx+9]]):
                count+=1
                sliced_index = ele_idx
                reference_indices.append(ele_idx)
        elif new_ele =='Refer' and ele_idx!= len(result_list)-2:
            if result_list[ele_idx+1][0] == 'ences':
                count+=1
                sliced_index = ele_idx
                reference_indices.append(ele_idx)

        elif new_ele =='r' and ele_idx!= len(result_list)-2:
            if result_list[ele_idx+1][0].lower() == 'eferences':
                count+=1
                sliced_index = ele_idx
                reference_indices.append(ele_idx)
    # print("count:", count)
    my_dict['ref'][0] = count
    my_dict['ref'][1] = reference_indices


    # ackn_indices = [i for i, (curr_text, curr_path) in enumerate(result_list) if remove_hashtags_colons(curr_text) in ['Acknowledgments', 'ACKNOWLEDGMENTS', 'Acknowledgment', 'ACKNOWLEDGMENT','Acknowledgements', 'ACKNOWLEDGEMENTS', 'Acknowledgement', 'ACKNOWLEDGEMENT', "ACKN", "Acknowledgments.", 'acknowledgements', 'acknowledgement', 'acknowledgments', 'acknowledgment']]
    # my_dict['ackn'][0] = len(ackn_indices)
    # if len(ackn_indices) > 0:
    #     my_dict['ackn'][1] = min(ackn_indices)

    bib_indices = [i for i, (curr_text, curr_path) in enumerate(result_list) if remove_hashtags_colons(curr_text) in ['Bibliography', 'BIBLIOGRAPHY']]

    my_dict['bib'][0] = len(bib_indices)
    if len(bib_indices) > 0:
        my_dict['bib'][1] = bib_indices

    coi_indices = [i for i, (curr_text, curr_path) in enumerate(result_list) if "conflict of interest" in remove_hashtags_colons(curr_text).lower()]

    my_dict['coi'][0] = len(coi_indices)
    if len(coi_indices) > 0:
        my_dict['coi'][1] = coi_indices

    min_index = len(result_list)
    for _, (curr_count, curr_indices) in my_dict.items():
        if not curr_indices:
            continue
        tmp_min_index = min(curr_indices)
        if curr_count == 1 and tmp_min_index < min_index:
            min_index = tmp_min_index
    # print(my_dict)

    return result_list[:min_index],my_dict

def filter_above_abstract(results_list):
    abstract_index = None
    for ele_idx, ele in enumerate(results_list):
        if remove_punctuations(ele[0]) == 'Abstract':
            abstract_index = ele_idx
            break
    if abstract_index is None:
        return results_list
    elif abstract_index / len(results_list) >= 0.25:
        return results_list
    else:
        return results_list[abstract_index:]
#or 'similar articles' in remove_hashtags_colons(ele[0]).lower()
def remove_similar_clinical_trials_non_ichgcp(results_list):
    similar_clinical_trial_index = None
    for ele_idx, ele in enumerate(results_list):
        if 'similar trials' in ele[0].lower() or ele[0] == "## Similar articles":
            similar_clinical_trial_index = ele_idx
            break
    if similar_clinical_trial_index is None:
        return results_list
    else:
        return results_list[:similar_clinical_trial_index]

def remove_similar_clinical_trials_ichgcp(results_list):
    similar_clinical_trial_index = None
    for ele_idx, ele in enumerate(results_list):
        if "clinical trials on" in ele[0].lower():
            similar_clinical_trial_index = ele_idx
            break
    if similar_clinical_trial_index is None:
        return results_list
    else:
        return results_list[:similar_clinical_trial_index]

def remove_pubmed_tail(results_list):
    coi_index = None
    for ele_idx, ele in enumerate(results_list):
        if "## conflict of interest statement" in ele[0].lower() or "## comment in" in ele[0].lower():
            coi_index = ele_idx
            break
    if coi_index is None:
        return results_list
    else:
        return results_list[:coi_index]

def remove_footer(results_list):
    footer_index = None
    for ele_idx, ele in enumerate(results_list):
        if "## Mayo Clinic Footer" in ele[0]:
            footer_index = ele_idx
            break
    if footer_index is None:
        return results_list
    else:
        return results_list[:footer_index]

def remove_newsletter(results_list):
    # Only works for multiple references, otherwise doesn't make a difference
    references_indices = [idx for idx, ele in enumerate(results_list) if remove_punctuations(ele[0]) in ['eferences', 'REFERENCES', 'References', 'Referenc', 'IRODAL', 'ЛИТЕР', 'Literatura','Reference(s)',"Reference"]]
    newsletter_indices = [idx for idx, ele in enumerate(results_list) if '### Newsletter' in ele[0]]
    if len(references_indices) == 0 or len(newsletter_indices) == 0:
        return results_list
    if max(references_indices) < min(newsletter_indices):
        return results_list[:min(newsletter_indices)]
    else:
        return results_list

def filter_withpower_junk(results_list):
    junk_indices = [idx for idx, ele in enumerate(results_list) if '### Other People Viewed' == ele[0]]
    if len(junk_indices) == 0:
        return results_list
    else:
        return results_list[:min(junk_indices)]

def filter_academia_e(results_list):
    references_indices = [idx for idx, ele in enumerate(results_list) if '## Related papers' == ele[0]]
    if len(references_indices) == 0:
        return results_list
    else:
        return results_list[:min(references_indices)]

def filter_marketing_slogans(results_list):
    keyword_list = ['### YOU MIGHT ALSO LIKE', 'You might also like', 'You might also like...', '#### You Might Also Like', '#### You Might Also Like', '### You Might Also Like', 'FAQ', '## You have questions. We have answers.', 'Related Posts', '### Want to learn more? Explore our resources', 'Related Content', '### Related', '## Related', 'Similar works', '## More articles by BioSyngen', '## More news from Yale Medicine', '## Recommended reading', '## Diese Studie war nicht für Sie geeignet?', 'Related Topics', 'Related news', 'Published Research Related to This Trial', 'More about research at Mayo Clinic', 'ESMO Congress', 'Related Articles', 'Recent news', 'Also on EndoNews', 'Recommended reading', 'More on Health', 'AstraZeneca overview', '### Recent CL Answers', 'More Science Update', 'MORE ON THIS TOPIC', 'Also from this source', 'ABOUT ASTRO', 'Related Resources']
    new_keyword_list = ['Sign Up for Email', 'Add topic to email alerts', '### Read this next', '### Share Your Story', '### Request an Appointment', '### Donate Blood', '### Subscribe to Cancer Frontline', '## Related resources', 'MORE ON THIS TOPIC', '### Related Coverage', 'Recommended Reading', '## Continue Reading', 'Closed Trials', 'Related clinical guidance', '## Recommended Stories', '## Continue Reading', '### Related pages']
    keyword_list += new_keyword_list
    keyword_list = [remove_hashtags_colons(ele).lower() for ele in keyword_list]
    keyword_set = set(keyword_list)

    # Case sensitive
    start_with_list = ['More News From']
    for ele_idx in range(len(results_list)-1, -1, -1):
        ele = results_list[ele_idx]

        if remove_hashtags_colons(ele[0]).lower() in keyword_set:
            if ele_idx / len(results_list) >= 0.50:
                return results_list[:ele_idx]
            else:
                break
        else:
            for t in start_with_list:
                if remove_hashtags_colons(ele[0]).startswith(t) and ele_idx / len(results_list) >= 0.50:
                    return results_list[:ele_idx]
    return results_list

def normalize_to_url_html_map(obj):
    """
    Accepts either:
      1) {'scraped_results_batch': { url: {content: str, method: str}, ... }}
         -> keeps content IFF method != 'failed'
      2) { url: html_str }  (legacy) OR { url: {content, method?} } mixed
    Returns: dict[url] = html_str  (only URLs we should process)
    """
    url_html = {}

    if isinstance(obj, dict) and 'scraped_results_batch' in obj:
        batch = obj.get('scraped_results_batch', {})
        if isinstance(batch, dict):
            for url, rec in batch.items():
                if not isinstance(rec, dict):
                    continue
                method = (rec.get('method') or '').lower()
                if method == 'failed':
                    continue
                content = rec.get('content')
                if content:
                    url_html[url] = content
        return url_html

    return url_html
