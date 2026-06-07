# CT Open

Open-access benchmark and evaluation framework for **clinical trial outcome prediction**.

CT Open evaluates whether models can predict clinical trial outcomes **before results become publicly available**. The benchmark uses time-stamped train/test splits and a decontamination pipeline to reduce the risk that models rely on already-public trial results.

## Overview

Clinical trial outcome prediction is a high-stakes forecasting problem with consequences for patients, clinicians, pharmaceutical companies, and investors. CT Open focuses on prediction from publicly available trial information available before a fixed cutoff date. Trials with evidence of public results before the cutoff are excluded through an automated filtering and verification pipeline.

CT Open supports three question classes:

* **Superiority**: whether the treatment arm shows a statistically significant improvement over the comparator arm
* **Comparative Effect**: whether one arm is significantly better, worse, or not different from the comparator
* **Endpoint**: whether the endpoint is met, or whether at least one arm meets the endpoint

## Repository Goals

This repository supports:

* construction of contamination-resistant clinical trial forecasting benchmarks
* preparation of train and time-stamped test sets
* filtering of trials with public outcome evidence before the benchmark cutoff
* answer generation and answer verification
* evaluation of prompt-only, retrieval-augmented, agentic, and traditional ML baselines

## Repository Structure

```text
ct-open/
├── README.md
├── datasets/
│   ├── Winter_2025.pickle
│   └── Summer_2025.pickle
├── pipelines/
│   ├── create_pipeline_data.py
│   ├── precheck_prompt_creation.py
│   ├── generation.py
│   ├── gemini_generation.py
│   ├── brave_api.py
│   ├── brave_search_single_query.py
│   ├── websearch.py
│   ├── zenrows_scraping.py
│   ├── request_scraping.py
│   ├── agent_zenrows.py
│   ├── parse_html_page_multi_thread.py
│   ├── parse_html_page_utils.py
│   ├── get_html_date.py
│   ├── process_dates.py
│   ├── check_match_round1.py
│   ├── check_match_round2.py
│   ├── insert_to_db.py
│   ├── replay_states.py
│   ├── filtered_trials_nctid2trial.pickle
│   ├── error_points.pickle
│   ├── winter2025_round1_prompts.pickle
│   ├── winter2025_websearch_round1_prompts.pickle
│   └── run_brave_experiments.sh
└── evaluations/
    └── evaluation.py
```

The benchmark has a **static component** with a training set and two time-stamped test sets, and a **dynamic component** with recurring benchmark releases.

## Datasets

CT Open includes three mutually disjoint trial sets:

* **Train**: 7,292 unique trials and 15,444 total questions
* **Winter 2025**: 314 unique trials and 605 total questions
* **Summer 2025**: 240 unique trials and 857 total questions

Since github can't upload file that exceed 25MB, so we devide Train data into 3 parts, please merge them when you want to use the Train data.

The static benchmarks are time-stamped:

* **Winter 2025** uses a cutoff date of **February 1, 2025**
* **Summer 2025** uses a cutoff date of **September 1, 2025**

Trials in each benchmark had no identified public results before the corresponding cutoff date.

## Benchmark Construction

### 1. Decontamination pipeline

The decontamination pipeline filters out trials with evidence of public results before a benchmark cutoff. It combines:

* LLM-based web search
* search-engine-based retrieval
* webpage scraping and downloadable file processing
* publication date extraction
* database search over sources such as PubMed, PMC, bioRxiv, and medRxiv
* two-round verification to confirm both trial identity and the presence of reported results

Manual review in the paper estimates the decontamination accuracy to be at least **98%** under a conservative evaluation.

### 2. Answer generation pipeline

After decontamination, CT Open uses a multi-stage pipeline to check whether result documents are sufficient to answer generated benchmark questions. Questions that cannot be answered with sufficient certainty are removed. The paper estimates answer generation accuracy to be at least **99%** under a conservative evaluation.

## Evaluation Settings

This repository supports evaluation of:

* **Prompt-based LLMs**
* **LLMs with retrieval-augmented generation**
* **Agentic LLMs**
* **Traditional machine learning baselines**
* **Neural network baselines**

In the agentic setting, the model can iteratively search, open retrieved pages, summarize findings, and produce a final prediction.

## Challenge Schedule

CT Open is designed as a recurring benchmark with four annual challenge cycles:

* **Winter Open**: December to March
* **Spring Open**: March to June
* **Summer Open**: June to September
* **Fall Open**: September to December

Participants submit predictions before a challenge window begins. Evaluation is performed on trials whose outcomes became public during the challenge window and were not public beforehand.

## Repository Contents

The repository includes:

* processed benchmark datasets
* metadata for trials, endpoints, and study arms
* intermediate retrieval and filtering artifacts
* prompt templates for search, matching, verification, and answer checking
* training and evaluation pipelines
* benchmark metrics and leaderboard generation code

## Intended Use

This project is intended for research on:

* clinical trial forecasting
* contamination-resistant benchmark design
* retrieval and agentic reasoning for scientific prediction
* evaluation of LLM and non-LLM approaches on time-stamped, open-world tasks

## Notes

* The benchmark uses publicly available aggregate trial information, not patient-level data.
* Trial sets for Train, Winter 2025, and Summer 2025 are mutually disjoint.
* This repository is organized to match the main components of the CT Open benchmark.

## Citation

If you use this repository or build on the benchmark design, please cite the CT Open paper.

```bibtex
@article{ctopen2026,
  title={CT Open: An Open-Access, Uncontaminated, Live Platform for the Open Challenge of Clinical Trial Outcome Prediction},
  author={Anonymous authors},
  journal={Under review as a conference paper at COLM 2026},
  year={2026}
}
```
