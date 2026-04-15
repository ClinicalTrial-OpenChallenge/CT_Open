import argparse
import pickle
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

from generation import generate


PROMPT = """Given some information about a clinical trial, output a one-sentence single search query without logical operators that searches for relevant outcomes on the web for the given clinical trial after the keyword "OUTPUT:"

Clinical trial information:
"""


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, payload) -> None:
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def build_prompt_dict(data, websearch_results_all):
    prompt_dict = {}

    for nctid in websearch_results_all:
        trial = data[nctid]
        tmp = []
        brief_title = trial.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", "")
        official_title = trial.get("protocolSection", {}).get("identificationModule", {}).get("officialTitle", "")
        brief_summary = trial.get("protocolSection", {}).get("descriptionModule", {}).get("briefSummary", "")
        detailed_description = trial.get("protocolSection", {}).get("descriptionModule", {}).get("detailedDescription", "")
        outcomes = trial.get("protocolSection", {}).get("outcomesModule", "")
        eligibility_criteria = trial.get("protocolSection", {}).get("eligibilityModule", {}).get("eligibilityCriteria", "")

        if brief_title:
            tmp.append("brief title: " + brief_title + "\n")
        if official_title:
            tmp.append("official title: " + official_title + "\n")
        if brief_summary:
            tmp.append("brief summary: " + brief_summary + "\n")
        if detailed_description:
            tmp.append("detailed description: " + detailed_description + "\n")
        if outcomes:
            tmp.append("outcomes: " + str(outcomes) + "\n")
        if eligibility_criteria:
            tmp.append("eligibility criteria: " + eligibility_criteria + "\n")

        prompt_dict[nctid] = PROMPT
        for value in tmp:
            prompt_dict[nctid] += value

    return prompt_dict


def extract_llm_rewrite_data(results):
    llm_rewrite_data = {}
    missing_output_nctids = []

    for nctid, result in results.items():
        if not result or "OUTPUT:" not in result:
            missing_output_nctids.append(nctid)
            continue
        llm_rewrite_data[nctid] = result.split("OUTPUT:", 1)[1].strip()

    return llm_rewrite_data, missing_output_nctids


def build_pipeline_data(data, websearch_results_all, llm_rewrite_data):
    pipeline_data = {}
    run_error_points = {}

    for nctid in websearch_results_all:
        trial = data[nctid]
        base_entry = {
            "ele": trial,
            "earliest_date": None,
            "search_round": 0,
            "brief_title": trial.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", ""),
            "nct_id": nctid,
        }

        llm_rewrite = llm_rewrite_data.get(nctid)
        if llm_rewrite is None:
            run_error_points[nctid] = base_entry
            continue

        pipeline_data[nctid] = {
            **base_entry,
            "llm_rewrite": llm_rewrite,
        }

    return pipeline_data, run_error_points


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial-data-path",
        type=Path,
        default=BASE_DIR / "filtered_trials_nctid2trial.pickle",
        help="Path to the trial metadata pickle.",
    )
    parser.add_argument(
        "--websearch-results-path",
        type=Path,
        default=BASE_DIR / "mar13_websearch_results.pickle",
        help="Path to the websearch results pickle.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=BASE_DIR / "mar13_pipeline_data.pickle",
        help="Path to save the pipeline data pickle.",
    )
    parser.add_argument(
        "--run-error-output-path",
        type=Path,
        default=BASE_DIR / "error_points.pickle",
        help="Path to save points that did not get an llm_rewrite.",
    )
    parser.add_argument("--model", type=str, default="gpt-5-2025-08-07")
    parser.add_argument("--max-completion-tokens", type=int, default=20000)
    parser.add_argument("--reasoning", type=str, default="medium")
    parser.add_argument("--verbosity", type=str, default="low")
    parser.add_argument("--timeout", type=int, default=600)
    return parser.parse_args()


def main():
    args = parse_args()

    data = load_pickle(args.trial_data_path)
    websearch_results_all = load_pickle(args.websearch_results_path)

    prompt_dict = build_prompt_dict(data, websearch_results_all)
    results, timeout, cost = generate(
        prompt_dict=prompt_dict,
        model=args.model,
        max_completion_tokens=args.max_completion_tokens,
        reasoning=args.reasoning,
        verbosity=args.verbosity,
        timeout=args.timeout,
    )

    llm_rewrite_data, missing_output_nctids = extract_llm_rewrite_data(results)
    pipeline_data, run_error_points = build_pipeline_data(data, websearch_results_all, llm_rewrite_data)

    for nctid in timeout:
        if nctid not in run_error_points:
            trial = data[nctid]
            run_error_points[nctid] = {
                "ele": trial,
                "earliest_date": None,
                "search_round": 0,
                "brief_title": trial.get("protocolSection", {}).get("identificationModule", {}).get("briefTitle", ""),
                "nct_id": nctid,
            }

    save_pickle(args.output_path, pipeline_data)
    save_pickle(args.run_error_output_path, run_error_points)

    print(f"Loaded {len(data)} trials")
    print(f"Loaded {len(websearch_results_all)} websearch result entries")
    print(f"Created {len(pipeline_data)} pipeline entries")
    print(f"Saved pipeline data to {args.output_path}")
    print(f"Saved run error points to {args.run_error_output_path}")
    print(f"Timed out prompts: {len(timeout)}")
    print(f"Missing OUTPUT marker: {len(missing_output_nctids)}")
    print(f"Total cost: {cost}")


if __name__ == "__main__":
    main()
