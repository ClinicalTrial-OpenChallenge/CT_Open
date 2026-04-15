import argparse
import pickle
import re
from datetime import datetime, date
from generation import generate

# ---------------------------------------------------------------------------
# Defaults (non-input paths)
# ---------------------------------------------------------------------------
DATE_PATTERN = re.compile(r"\bDATE:\s*(\d{4}-\d{2}-\d{2})\b")
CUTOFF = "2025-01-31"

PROMPT_TEMPLATE = (
    "Given a url, get the earliest publish date of the url's main content. "
    "Output the extracted date after the keyword \"DATE:\" in the format 'YYYY-MM-DD'.\n\n"
    "URL: {url}"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, path: str):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def parse_date_or_none(text, today=None):
    """Extract a valid YYYY-MM-DD date string from model output, or None."""
    if not isinstance(text, str):
        return None
    m = DATE_PATTERN.search(text)
    if not m:
        return None
    date_str = m.group(1)
    if date_str == "0000-00-00":
        return None
    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError:
        return None
    if today is None:
        today = date.today()
    if parsed_date > today:
        return None
    return date_str


def parse_args():
    p = argparse.ArgumentParser(
        description="Merge round2 websearch pickles and fill missing dates via generate()."
    )
    p.add_argument(
        "--round2-result-gemini",
        required=True,
        help="Pickle: round2 gemini (before cutoff) websearch parsed.",
    )
    p.add_argument(
        "--round2-result-gpt",
        required=True,
        help="Pickle: round2 GPT (before cutoff) websearch parsed.",
    )
    p.add_argument(
        "--round2-result-gemini-after",
        required=True,
        help="Pickle: round2 gemini (after cutoff) websearch parsed.",
    )
    p.add_argument(
        "--round2-result-gpt-after",
        required=True,
        help="Pickle: round2 GPT (after cutoff) websearch parsed.",
    )
    p.add_argument(
        "--model-output",
        default="data/Mar11_none_date_res_test.pickle",
        help="Path to write intermediate model responses pickle.",
    )
    p.add_argument(
        "--final-output",
        default="data/Mar11_combined_res_final_test.pickle",
        help="Path to write final merged pickle.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pickle_inputs = {
        "round2_result_gemini": args.round2_result_gemini,
        "round2_result_gpt": args.round2_result_gpt,
        "round2_result_gemini_after": args.round2_result_gemini_after,
        "round2_result_gpt_after": args.round2_result_gpt_after,
    }

    # ---------------------------------------------------------------------------
    # 1. Load sources
    # ---------------------------------------------------------------------------
    sources = {name: load_pickle(path) for name, path in pickle_inputs.items()}

    # ---------------------------------------------------------------------------
    # 2. Merge all sources into combined_res
    # ---------------------------------------------------------------------------
    combined_res: dict = {}

    for source in sources.values():
        for res_key, urls in source.items():
            for url, data in urls.items():
                if res_key not in combined_res:
                    combined_res[res_key] = {}
                if url in combined_res[res_key]:
                    existing_date = combined_res[res_key][url]["date"]
                    new_date = data["date"]
                    if new_date != existing_date:
                        combined_res[res_key][url]["date"] = min(
                            d for d in (existing_date, new_date) if d is not None
                        )
                else:
                    combined_res[res_key][url] = data

    # ---------------------------------------------------------------------------
    # 3. Build prompt list & prune already-outdated entries
    # ---------------------------------------------------------------------------
    combined_res_copy = {k: dict(v) for k, v in combined_res.items()}

    prepare_for_prompt: list[tuple[str, str]] = []
    pruned_keys: set = set()

    for nctid, url_map in combined_res.items():
        prune = False
        for url, data in url_map.items():
            d = data["date"]
            if d is None or d == "2025":
                prepare_for_prompt.append((nctid, url))
            elif d <= CUTOFF and d != "2025":
                prune = True
                break
        if prune:
            pruned_keys.add(nctid)

    for k in pruned_keys:
        combined_res_copy.pop(k, None)

    # ---------------------------------------------------------------------------
    # 4. Call model for entries with missing / ambiguous dates
    # ---------------------------------------------------------------------------
    prompt_dict = {
        (nctid, url): PROMPT_TEMPLATE.format(url=url)
        for nctid, url in prepare_for_prompt
        if nctid in combined_res_copy and "vertexai" not in url
    }

    res, time_o, cost = generate(
        prompt_dict=prompt_dict,
        model="gpt-5-2025-08-07",
        max_completion_tokens=20000,
        reasoning="medium",
        verbosity="low",
        timeout=600,
        web_search=True,
    )

    save_pickle(res, args.model_output)

    # ---------------------------------------------------------------------------
    # 5. Merge model-returned dates back
    # ---------------------------------------------------------------------------
    for (nctid, url), value in res.items():
        if nctid in combined_res_copy:
            combined_res_copy[nctid][url]["date"] = parse_date_or_none(value)

    # ---------------------------------------------------------------------------
    # 6. Final prune: drop entries still missing dates (from original sources)
    #    or with dates before cutoff
    # ---------------------------------------------------------------------------
    round2_result_gpt = sources["round2_result_gpt"]
    round2_result_gemini = sources["round2_result_gemini"]

    final_res = {k: dict(v) for k, v in combined_res_copy.items()}

    for nctid, url_map in combined_res_copy.items():
        drop = False
        for url, data in url_map.items():
            d = data["date"]
            if d is None:
                if (nctid in round2_result_gemini and url in round2_result_gemini[nctid]) or \
                   (nctid in round2_result_gpt and url in round2_result_gpt[nctid]):
                    drop = True
                    break
            elif d <= CUTOFF:
                drop = True
                break
        if drop:
            final_res.pop(nctid, None)

    # ---------------------------------------------------------------------------
    # 7. Save final output (same pickle format as combined_res_copy_copy)
    # ---------------------------------------------------------------------------
    save_pickle(final_res, args.final_output)
    print(f"Saved {len(final_res)} entries to {args.final_output}")


if __name__ == "__main__":
    main()
