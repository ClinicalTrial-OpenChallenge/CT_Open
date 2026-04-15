import argparse
import pickle
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_TRIAL_DATA_PATH = BASE_DIR / "filtered_trials_nctid2trial.pickle"
DEFAULT_EXISTING_PROMPT_PATH_1 = BASE_DIR / "winter2025_websearch_round1_prompts.pickle"
DEFAULT_EXISTING_PROMPT_PATH_2 = BASE_DIR / "winter2025_round1_prompts.pickle"
DEFAULT_OUTPUT_PATH = BASE_DIR / "precheck_prompt_dict.pickle"

INSTRUCTION = """
You are given the NCTID of a clinical trial. You are also given a clinical trial's title, description, and outcome measures. Search the internet to find any mention of the clinical trial's results (it can be interim results or any indication of results on any of its outcome measures) from any webpage such as news, company statements, papers (e.g. pubmed, pmc, bioRxiv, medRxiv, clinicaltrials.gov), blogs and online forums. When you find one webpage that mentions this exact clinical trial's NCTID and mentions at least some interim results of this exact clinical trial, stop the search all together. If you find such a webpage, output "yes" after the keyword "OUTPUT:"

Note, if you still cannot find any webpage with any results (good or bad) for this specific clinical trial, output "no" after the keyword "OUTPUT:"

NCTID:
{nctid}

Clinical trial description:
{ct_description}

Clinical trial outcome measures:
{ct_outcome_measures}
""".strip()


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path: Path, payload) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def get_remaining_nctids(nct2module, existing_prompt_dict_1, existing_prompt_dict_2):
    nctid_2430 = set(existing_prompt_dict_1.keys()) | set(existing_prompt_dict_2.keys())
    return list(set(nct2module.keys()) - nctid_2430)


def build_prompt_dict(nct2module, nctid_list):
    prompt_dict = {}

    for nctid in nctid_list:
        ele = nct2module[nctid]
        curr_brief_title = ele.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", "")
        curr_official_title = ele.get("protocolSection", {}).get("identificationModule", {}).get("officialTitle", "")

        curr_brief_summary = ele.get("protocolSection", {}).get("descriptionModule", {}).get("briefSummary", "")
        curr_detailed_description = ele.get("protocolSection", {}).get("descriptionModule", {}).get("detailedDescription", "")

        if len(curr_brief_title.split()) > len(curr_official_title.split()):
            my_title = curr_brief_title
        else:
            my_title = curr_official_title

        if len(curr_brief_summary.split()) > len(curr_detailed_description.split()):
            my_summary = curr_brief_summary
        else:
            my_summary = curr_detailed_description

        curr_outcome_measures = ele.get("protocolSection", {}).get("outcomesModule", {})
        curr_ct_description = f"Title: {my_title}\n\nSummary: {my_summary}"
        my_prompt = INSTRUCTION.format(
            nctid=nctid,
            ct_description=curr_ct_description,
            ct_outcome_measures=curr_outcome_measures,
        )
        prompt_dict[nctid] = my_prompt

    return prompt_dict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial-data-path",
        type=Path,
        default=DEFAULT_TRIAL_DATA_PATH,
        help="Path to the trial metadata pickle.",
    )
    parser.add_argument(
        "--existing-prompt-path-1",
        type=Path,
        default=DEFAULT_EXISTING_PROMPT_PATH_1,
        help="Path to the first existing prompt pickle.",
    )
    parser.add_argument(
        "--existing-prompt-path-2",
        type=Path,
        default=DEFAULT_EXISTING_PROMPT_PATH_2,
        help="Path to the second existing prompt pickle.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the generated prompt_dict pickle.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    nct2module = load_pickle(args.trial_data_path)
    existing_prompt_dict_1 = load_pickle(args.existing_prompt_path_1)
    existing_prompt_dict_2 = load_pickle(args.existing_prompt_path_2)

    sep_nctid_list = get_remaining_nctids(
        nct2module,
        existing_prompt_dict_1,
        existing_prompt_dict_2,
    )
    prompt_dict = build_prompt_dict(nct2module, sep_nctid_list)

    dump_pickle(args.output_path, prompt_dict)

    print(f"Loaded {len(nct2module)} trials")
    print(f"Found {len(sep_nctid_list)} remaining NCTIDs")
    print(f"Saved {len(prompt_dict)} prompts to {args.output_path}")


if __name__ == "__main__":
    main()
