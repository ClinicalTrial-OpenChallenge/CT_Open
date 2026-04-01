# CT Open

Open-access benchmark and evaluation framework for **clinical trial outcome prediction**.

This repository is designed around the CT Open setting: predicting clinical trial outcomes **before results become publicly available**, using contamination-resistant benchmark construction and time-stamped evaluation. CT Open includes a large-scale training set, two static benchmarks, and a recurring live evaluation setting built around quarterly challenge cycles.

## Overview

Clinical trial outcome prediction is a high-stakes forecasting problem with consequences for patients, clinicians, pharmaceutical companies, and investors. CT Open is built to evaluate whether AI systems can make useful predictions on trial outcomes **without relying on already-public results**. To support this, the benchmark uses an automated decontamination pipeline to exclude trials with publicly available outcome evidence prior to a benchmark cutoff date.

CT Open supports three question classes:

- **Superiority**: whether the treatment arm shows a statistically significant improvement over the comparator arm
- **Comparative Effect**: whether one arm is significantly better, worse, or not different from the comparator
- **Endpoint**: whether the endpoint is met, or whether at least one arm meets the endpoint

## Repository Goals

This repository is intended to support:

- benchmark construction for contamination-resistant clinical trial forecasting
- data preparation for train and time-stamped test sets
- decontamination and public-result filtering
- answer generation and answer verification
- evaluation of prompt-only, retrieval-augmented, agentic, and traditional ML baselines

## Suggested Repository Structure

```text
ct-open/
├── README.md
├── datasets/
│   ├── train/
│   ├── winter_2025/
│   └── summer_2025/
├── pipelines/
│   ├── decontamination/
│   ├── answer_generation/
│   ├── trial_matching/
│   ├── result_verification/
│   ├── publication_date_extraction/
│   └── web_retrieval/
├── evaluations/
│   ├── prompt_based/
│   ├── rag/
│   ├── agentic/
│   ├── traditional_ml/
│   └── neural_models/
├── prompts/
│   ├── screening/
│   ├── web_search/
│   ├── verification/
│   └── answer_checking/
├── docs/
│   ├── benchmark_design/
│   ├── pipeline_overview/
│   └── evaluation_protocol/
└── outputs/
    ├── intermediate/
    ├── filtered_trials/
    ├── verified_results/
    └── metrics/
```

This layout reflects the paper’s main components while staying implementation-agnostic. The benchmark has a **static component** with a training set and two time-stamped test sets, and a **dynamic component** with recurring benchmark releases.

## Datasets

CT Open includes three mutually disjoint trial sets:

- **Train**: 7,292 unique trials and 15,444 total questions
- **Winter 2025**: 314 unique trials and 605 total questions
- **Summer 2025**: 240 unique trials and 857 total questions

The static benchmarks are time-stamped:

- **Winter 2025** uses a cutoff date of **February 1, 2025**
- **Summer 2025** uses a cutoff date of **September 1, 2025**

By construction, trials in each benchmark had no publicly available results before the corresponding cutoff date.

## Benchmark Construction

### 1. Decontamination pipeline

The decontamination pipeline filters out trials with evidence of public results before a benchmark cutoff. It combines:

- LLM-based web search
- search-engine-based retrieval
- webpage scraping and downloadable file processing
- publication date extraction
- database search over sources such as PubMed, PMC, bioRxiv, and medRxiv
- two-round verification to confirm both trial identity and presence of reported results

The paper reports strong robustness for this pipeline, with manual review confirming at least **98% accuracy** under a conservative estimate.

### 2. Answer generation pipeline

After decontamination, CT Open uses a multi-stage pipeline to determine whether result documents are sufficient to answer generated benchmark questions. Questions that cannot be supported with sufficient certainty are removed. The paper reports at least **99% accuracy** for answer generation under conservative evaluation.

## Evaluation Settings

This repository is organized to support multiple modeling approaches discussed in the paper:

- **Prompt-based LLMs**
- **LLMs with retrieval-augmented generation**
- **Agentic LLMs**
- **Traditional machine learning baselines**
- **Neural network baselines**

The agentic setting follows an iterative search-and-browse workflow where the model can:
1. submit a new query,
2. request the full content of a previously retrieved URL,
3. summarize findings and update strategy,
4. produce a final answer.

## Challenge Schedule

CT Open is designed as a recurring benchmark with four annual challenge cycles:

- **Winter Open**: December to March
- **Spring Open**: March to June
- **Summer Open**: June to September
- **Fall Open**: September to December

Participants submit predictions before a challenge window begins, and evaluations are performed only on trials whose outcomes became public during that window but were not public beforehand.

## What This Repository Contains

Depending on your implementation, this repository can be used to host:

- processed benchmark datasets
- metadata for trials, endpoints, and study arms
- intermediate retrieval and filtering artifacts
- prompt templates for search, matching, verification, and answer checking
- training and evaluation pipelines
- benchmark metrics and leaderboard generation code

## Intended Use

This project is intended for research on:

- clinical trial forecasting
- contamination-resistant benchmark design
- retrieval and agentic reasoning for scientific prediction
- evaluation of LLM and non-LLM approaches on time-stamped, open-world tasks

## Notes

- The benchmark is designed around **publicly available aggregate trial information**, not patient-level data.
- Trial sets for train, Winter 2025, and Summer 2025 are **mutually disjoint**.
- The repository structure here is a **suggested organization** derived from the paper, not an attempt to reproduce unpublished internal filenames.

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
