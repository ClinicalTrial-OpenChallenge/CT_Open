#auto reload
from generation import generate
import pickle
import ast
import json
import re
from tqdm import tqdm
from typing import Dict, Any, List
import flatdict
from gemini_generation import run_batch


p = """
Find me results of this exact same Clinical Trial prior to this date: 2025/09/01.

Any mention of interim outcomes for the specific trial—such as success, failure, or signs of efficacy/safety—counts as reporting results for that trial.

However, if a webpage only reports results from related trials, studies, papers, or experiments (and not from this exact trial), then it is not reporting results for the specific trial.

Give me all the urls you found. Note you should only output urls that are reporting results for the specific trial, not urls that are related to the trial or a earlier phase of the trial.

Output format (must follow exactly)
- Start a new line with LIST: then a single valid Python list of strings.
- No extra commentary before or after the list.
- Example:
  LIST: [url1, url2, url3, url4, url5, ...]
"""

p2 = """
Task: Given some publicly available webpages that report outcomes for the exact clinical trial described below, extract the date and the result of the webpages for the exact clinical trial.

What counts as “results”: Any explicit reporting of outcomes (e.g., pCR, MFS, investigator/BICR assessments, efficacy signals, safety findings, success/failure of interim analyses) for this exact trial.

Dates: Extract the page's publication date. If multiple dates appear, use the one that reflects when the results were first posted. If you cannot determine a date, write “unknown”.

Result content: Under result:, add concise bullet points (ideally 1-3) summarizing the evidence of outcomes reported on that page. Prefer specific numbers (e.g., pCR %, MFS hazard ratios), time frames, and clear efficacy/safety statements. Do not include commentary beyond what the page states.

Output format (must follow exactly; no extra text before or after):

url1: <url1>
date: <YYYY-MM-DD or unknown>
result:

<data point 1 demonstrating results>

<data point 2 demonstrating results>


url2: <url2>
date: <YYYY-MM-DD or unknown>
result:

<data point 1 demonstrating results>

<data point 2 demonstrating results>

(Repeat for all qualifying URLs.)

Trial to match (must be this exact study):
{trial}

URLs to process:
{urls}
"""


p_after = """
Find me results of this exact same Clinical Trial on or after this date: 2025/09/01.

Any mention of interim outcomes for the specific trial—such as success, failure, or signs of efficacy/safety—counts as reporting results for that trial.

However, if a webpage only reports results from related trials, studies, papers, or experiments (and not from this exact trial), then it is not reporting results for the specific trial.

Give me all the urls you found. Note you should only output urls that are reporting results for the specific trial, not urls that are related to the trial or a earlier phase of the trial.

Output format (must follow exactly)
- Start a new line with LIST: then a single valid Python list of strings.
- No extra commentary before or after the list.
- Example:
  LIST: [url1, url2, url3, url4, url5, ...]
"""

def to_url_list(value):
    """Coerce a value into a list of URLs from varied string formats."""
    if isinstance(value, list):
        return value

    s = "" if value is None else str(value).strip()

    # Normalize 'list:' prefix (any casing, optional spaces)
    if s.lower().startswith("list:"):
        s = s.split(":", 1)[1].strip()

    # Treat explicit empties as []
    if re.fullmatch(r"\[\s*\]", s) or re.fullmatch(r"list:\s*\[\s*\]", s, flags=re.IGNORECASE):
        return []

    # Try to isolate the bracketed portion if present
    m = re.search(r"\[.*\]$", s, flags=re.DOTALL)
    s_bracket = m.group(0) if m else s

    # Try Python literal parsing
    try:
        parsed = ast.literal_eval(s_bracket)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Try JSON parsing
    try:
        parsed = json.loads(s_bracket)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Fallback: extract URLs
    urls = re.findall(r'https?://\S+?(?=["\]\s,>)}])', s)
    return urls

def normalize_output(output_dict):
    """Return a dict {key: list_of_urls}, keeping keys even when empty."""
    result = {}
    for key, value in output_dict.items():
        result[key] = to_url_list(value)
    return result

import re
from typing import Dict, Any, List


def _normalize_date(raw_date: str) -> str | None:
    """Return the date string only if it matches YYYY-MM-DD, otherwise None."""
    raw_date = raw_date.strip()
    if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', raw_date):
        return raw_date
    return None


def _split_results(results_block: str) -> List[str]:
    """
    Turn a free-form 'result:' block into a clean list of bullet points.
    Accepts lines starting with '-', '•', or '*', and also splits on '- ' inside the block.
    """
    text = results_block.strip()
    if not text:
        return []

    lines = []
    for raw in text.splitlines():
        s = raw.strip()
        if s:
            lines.append(s)

    bullets = []
    for line in lines:
        inline_parts = re.split(r'\s(?=-\s|\*\s|•\s)', line)
        for part in inline_parts:
            part = part.strip()
            part = re.sub(r'^(?:-\s+|\*\s+|•\s+)', '', part).strip()
            if part:
                bullets.append(part)

    if not bullets and '- ' in text:
        for piece in text.split('- '):
            piece = piece.strip()
            if piece:
                bullets.append(piece)

    seen = set()
    out = []
    for b in bullets:
        if b not in seen:
            seen.add(b)
            out.append(b)
    return out


def parse_output_dict(output_dict2: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Parse values like:
      "url1: https://... date: 2025-01-14 result: - a ... - b ...
       url2: https://... date: 2025-05-03 result: - c ..."
    Into:
      {
        NCTxxxxxxx: {
          "https://...": {"date": "2025-01-14", "results": ["a ...", "b ..."]},
          "https://...": {"date": "2025-05-03", "results": ["c ..."]}
        },
        ...
      }
    """
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}

    segment_re = re.compile(
        r'url\d*\s*:\s*(\S+)'               # 1) URL
        r'.*?date\s*:\s*(\S+)'              # 2) date (any token — normalized later)
        r'.*?result\s*:\s*(.*?)'            # 3) result block
        r'(?=(?:\burl\d*\s*:)|\Z)',         # lookahead to next urlX: or end
        flags=re.IGNORECASE | re.DOTALL
    )

    for nct_id, raw in output_dict2.items():
        val = '' if raw is None else str(raw)
        if not val.strip() or val.strip() == '[]':
            continue

        matches = list(segment_re.finditer(val))

        if not matches:
            url = None
            date = None
            results_block = None

            m_url = re.search(r'\burl\d*\s*:\s*(\S+)', val, flags=re.IGNORECASE)
            if m_url:
                url = m_url.group(1).strip()

            m_date = re.search(r'\bdate\s*:\s*(\S+)', val, flags=re.IGNORECASE)
            if m_date:
                date = _normalize_date(m_date.group(1))

            m_result = re.search(r'\bresult\s*:\s*(.*)\Z', val, flags=re.IGNORECASE | re.DOTALL)
            if m_result:
                results_block = m_result.group(1)

            if url and (date or results_block):
                out.setdefault(nct_id, {})
                out[nct_id][url] = {
                    "date": date,
                    "results": _split_results(results_block or "")
                }
            continue

        for m in matches:
            url = m.group(1).strip()
            date = _normalize_date(m.group(2))
            results_block = m.group(3)
            results = _split_results(results_block or "")

            if url:
                out.setdefault(nct_id, {})
                out[nct_id][url] = {"date": date, "results": results}

    return out




def main(args):
    # aug15_36k_clinical_trials.pickle
    with open(args.all_36k_trials_path, 'rb') as f:
        trials_data = pickle.load(f)


    with open(args.input_nctid_file,'rb') as f: 
        total_list_nctid_200 = pickle.load(f)

    model_type = args.model_type

    prompt_dict = {}
    for nctid in total_list_nctid_200:
        temp = ""
        for k, d in trials_data[nctid]['protocolSection'].items():

            if k == "eligibilityModule":
                continue
            fd = flatdict.FlatDict(d, delimiter=": ")
            for key,value in fd.items():
                if (type(key) == str and 'date' in key.lower()) or (type(value) == str and 'date' in value.lower()):
                    continue
                temp += f"{key}: {value}\n"
        prompt_dict[nctid] = p + '\n' + temp 

    # with open("../../generation/prompts/oct21_before_0201_round1_prompt.pickle","wb") as f:
    #     pickle.dump(prompt_dict,f)
    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round1_prompt.pickle","wb") as f:
        pickle.dump(prompt_dict,f)

    # output_dict, timed_out_prompts, total_cost = generate(prompt_dict = prompt_dict, model = 'gpt-5.2-2025-12-11', max_completion_tokens=15000,reasoning="high", verbosity="high", timeout=600, web_search=True)
    # with open('/path/to/gpt_responses/oct20_sample200_web_search_result.pickle','wb') as f:
    #     pickle.dump(output_dict,f)
    
    if (model_type == 'gpt'):
        print("Using GPT-5.2-2025-12-11 with 60000 max completion tokens")
        print("Make sure you have enough tokens to complete the output, check the result when switching to a new model.")
        output_dict, timed_out_prompts, total_cost = generate(prompt_dict = prompt_dict, model = 'gpt-5.2-2025-12-11', max_completion_tokens=60000,reasoning="high", verbosity="high", timeout=600, web_search=True)
        if len(timed_out_prompts)>0:
            with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round1_web_search_result_timed_out_prompts.pickle",'wb') as f:
                pickle.dump(timed_out_prompts,f)
    else:
        print("Using Gemini-3-Pro with 60000 thinking budget")
        print("Make sure you have enough tokens to complete the output, check the result when switching to a new model.")
        output_dict, summary = run_batch(prompts=prompt_dict)
        print(f"Completed {len(output_dict)} items | "
            f"in={summary['totals']['input_tokens']} out={summary['totals']['output_tokens']} | "
            f"token_cost=${summary['totals']['token_cost_usd']:.4f} + "
            f"grounding=${summary['totals']['grounding_cost_usd']:.4f} "
            f"= total=${summary['totals']['cost_usd']:.4f}")

    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round1_web_search_result.pickle",'wb') as f:
        pickle.dump(output_dict,f)

    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round1_web_search_result.pickle",'rb')as f:
        output_dict = pickle.load(f)

    cleaned = normalize_output(output_dict)
    # with open('/path/to/data/oct17_sample_round1_parse.pickle','wb') as f:
    #     pickle.dump(cleaned,f)

    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round1_sample_parse.pickle",'wb') as f:
        pickle.dump(cleaned,f)



    prompt_dict2 = {}
    for nctid in tqdm(cleaned):
        if cleaned[nctid] == []:
            continue
        temp = ""
        for k, d in trials_data[nctid]['protocolSection'].items():

            if k == "eligibilityModule":
                continue
            fd = flatdict.FlatDict(d, delimiter=": ")
            for key,value in fd.items():
                if (type(key) == str and 'date' in key.lower()) or (type(value) == str and 'date' in value.lower()):
                    continue
                temp += f"{key}: {value}\n"
        prompt_dict2[nctid] = p2.format(trial=temp,urls=cleaned[nctid])
    # output_dict2, timed_out_prompts2, total_cost2 = generate(prompt_dict = prompt_dict2, model = 'gpt-5.2-2025-12-11', max_completion_tokens=15000,reasoning="high", verbosity="high", timeout=600, web_search=True)
    # with open('/path/to/gpt_responses/oct20_sample200_web_search_second_round_result.pickle','wb') as f:
    #     pickle.dump(output_dict2,f)


    if (model_type == 'gpt'):
        output_dict2, timed_out_prompts2, total_cost2 =  generate(prompt_dict = prompt_dict2, model = 'gpt-5.2-2025-12-11', max_completion_tokens=15000,reasoning="high", verbosity="high", timeout=600, web_search=True)
        if len(timed_out_prompts2)>0:
            with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round2_web_search_result_timed_out_prompts2.pickle",'wb') as f:
                pickle.dump(timed_out_prompts2,f)
    else:
        output_dict2, summary = run_batch(prompts=prompt_dict2)
        print(f"Completed {len(output_dict2)} items | "
            f"in={summary['totals']['input_tokens']} out={summary['totals']['output_tokens']} | "
            f"token_cost=${summary['totals']['token_cost_usd']:.4f} + "
            f"grounding=${summary['totals']['grounding_cost_usd']:.4f} "
            f"= total=${summary['totals']['cost_usd']:.4f}")

    
    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round2_web_search_result.pickle",'wb') as f:
        pickle.dump(output_dict2,f)

    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round2_web_search_result.pickle",'rb') as f:
        output_dict2 = pickle.load(f)


    parsed = parse_output_dict(output_dict2)
    # with open('/path/to/data/oct20_sample200_web_search_second_round_parsed.pickle','wb') as f:
    #     pickle.dump(parsed,f)
    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round2_web_search_parsed.pickle",'wb') as f:
        pickle.dump(parsed,f)

    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_before_0201_round2_web_search_parsed.pickle",'rb') as f:
        parsed = pickle.load(f)


    parsed_before_0201 = parsed
    remain_150 = set(total_list_nctid_200)-set(parsed_before_0201)




    # TODO: We will let the rest code into second py.file, remain_150 will become the next file's input.

    prompt_dict = {}
    for nctid in tqdm(remain_150):
        temp = ""
        for k, d in trials_data[nctid]['protocolSection'].items():

            if k == "eligibilityModule":
                continue
            fd = flatdict.FlatDict(d, delimiter=": ")
            for key,value in fd.items():
                if (type(key) == str and 'date' in key.lower()) or (type(value) == str and 'date' in value.lower()):
                    continue
                temp += f"{key}: {value}\n"
        prompt_dict[nctid] = p_after + '\n' + temp 


    # output_dict, timed_out_prompts, total_cost = generate(prompt_dict = prompt_dict, model = 'gpt-5.2-2025-12-11', max_completion_tokens=15000,reasoning="high", verbosity="high", timeout=600, web_search=True)
    # with open('/path/to/gpt_responses/oct20_round1_after0201_web_search_result.pickle','wb') as f:
    #     pickle.dump(output_dict,f)

    if (model_type == 'gpt'):
        output_dict, timed_out_prompts, total_cost =  generate(prompt_dict = prompt_dict, model = 'gpt-5.2-2025-12-11', max_completion_tokens=15000,reasoning="high", verbosity="high", timeout=600, web_search=True)
        if len(timed_out_prompts)>0:
            with open(f"{args.output_dir_to_websearch}/{args.exp_name}_round1_after0201_web_search_result_timed_out_prompts.pickle",'wb') as f:
                pickle.dump(timed_out_prompts,f)
    else:
        output_dict, summary = run_batch(prompts=prompt_dict)
        print(f"Completed {len(output_dict)} items | "
            f"in={summary['totals']['input_tokens']} out={summary['totals']['output_tokens']} | "
            f"token_cost=${summary['totals']['token_cost_usd']:.4f} + "
            f"grounding=${summary['totals']['grounding_cost_usd']:.4f} "
            f"= total=${summary['totals']['cost_usd']:.4f}")


    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_round1_after0201_web_search_result.pickle",'wb') as f:
        pickle.dump(output_dict,f)

    cleaned = normalize_output(output_dict)
    # with open('/path/to/data/oct20_sample_round1_after0201_parse.pickle','wb') as f:
    #     pickle.dump(cleaned,f)
    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_round1_after0201_parse.pickle",'wb') as f:
        pickle.dump(cleaned,f)

    prompt_dict2 = {}
    for nctid in tqdm(cleaned):
        if cleaned[nctid] == []:
            continue
        temp = ""
        for k, d in trials_data[nctid]['protocolSection'].items():

            if k == "eligibilityModule":
                continue
            fd = flatdict.FlatDict(d, delimiter=": ")
            for key,value in fd.items():
                if (type(key) == str and 'date' in key.lower()) or (type(value) == str and 'date' in value.lower()):
                    continue
                temp += f"{key}: {value}\n"
        prompt_dict2[nctid] = p2.format(trial=temp,urls=cleaned[nctid])


    # output_dict2, timed_out_prompts2, total_cost2 = generate(prompt_dict = prompt_dict2, model = 'gpt-5.2-2025-12-11', max_completion_tokens=15000,reasoning="high", verbosity="high", timeout=600, web_search=True)
    # with open('/path/to/gpt_responses/oct20_after_0201_web_search_second_round_result.pickle','wb') as f:
    #     pickle.dump(output_dict2,f)
    if (model_type == 'gpt'):
        output_dict2, timed_out_prompts2, total_cost2 =  generate(prompt_dict = prompt_dict2, model = 'gpt-5.2-2025-12-11', max_completion_tokens=15000,reasoning="high", verbosity="high", timeout=600, web_search=True)
        if len(timed_out_prompts2)>0:
            with open(f"{args.output_dir_to_websearch}/{args.exp_name}_round1_after0201_web_search_result_timed_out_prompts.pickle",'wb') as f:
                pickle.dump(timed_out_prompts2,f)
    else:
        output_dict2, summary = run_batch(prompts=prompt_dict2)
        print(f"Completed {len(output_dict2)} items | "
            f"in={summary['totals']['input_tokens']} out={summary['totals']['output_tokens']} | "
            f"token_cost=${summary['totals']['token_cost_usd']:.4f} + "
            f"grounding=${summary['totals']['grounding_cost_usd']:.4f} "
            f"= total=${summary['totals']['cost_usd']:.4f}")


    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_round2_after_0201_web_search_result.pickle",'wb') as f:
        pickle.dump(output_dict2,f)

    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_round2_after_0201_web_search_result.pickle",'rb') as f:
        output_dict2 = pickle.load(f)


    # parsed = parse_output_dict(output_dict2)
    # with open('/path/to/data/oct20_after_0201_web_search_second_round_parsed.pickle','wb') as f:
    #     pickle.dump(parsed,f)
    parsed = parse_output_dict(output_dict2)
    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_round2_after_0201_web_search_parsed.pickle",'wb') as f:
        pickle.dump(parsed,f)
    result_dict = {}
    
    
    for nctid in parsed:
        for url in parsed[nctid]:
            result_dict[(nctid,url,0)] = '\n'.join(parsed[nctid][url]['results'])

    with open(f"{args.output_dir_to_websearch}/{args.exp_name}_final_result_dict.pickle",'wb') as f:
        pickle.dump(result_dict,f)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="pipeline match by gpt before 0201 round 2.")
    parser.add_argument('--exp_name', type=str, required=True,  help='path to save the websearch results.')
    parser.add_argument('--all_36k_trials_path', type=str, required=True, help='path to load the file of all 36k clinical trials.')
    parser.add_argument('--input_nctid_file', type=str, required=True,  help='path to save the websearch results.')
    parser.add_argument('--output_dir_to_websearch', type=str, required=True, help='directory to save the data that will be used in round2_before_0201.')
    parser.add_argument('--model_type', type=str, required=True,help='choose your model type: gpt or gemini.')

    # Add any arguments you want to parse here
    args = parser.parse_args()
    # create the output directory if it does not exist
    import os
    os.makedirs(args.output_dir_to_websearch, exist_ok=True)
    main(args)
# python websearch.py --exp_name mar12_1267_nctids_gpt --all_36k_trials_path ./data/filtered_trials_nctid2trial.pickle --input_nctid_file ./september_benchmark/mar12_starting_nctids_sep_benchmark.pickle --output_dir_to_websearch ./september_benchmark/websearch_results --model_type gpt