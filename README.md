<div align="center">
<h1>CT Open: An Open-Access, Uncontaminated, Live Platform for the Open Challenge of Clinical Trial Outcome Prediction (COLM 2026)</h1>

<p>
<b>
Jianyou Wang<sup>1*</sup>, Youze Zheng<sup>1*</sup>, Longtian Bao<sup>1*</sup>, Hanyuan Zhang<sup>1*</sup><br>
Qirui Zheng<sup>1</sup>, Yuhan Chen<sup>1</sup>, Yang Zhang<sup>1</sup>, Matthew Feng<sup>1</sup><br>
Maxim Khan<sup>3</sup>, Aditya K. Sehgal<sup>3</sup>, Christopher D. Rosin<sup>3</sup><br>
Ramamohan Paturi<sup>1</sup>, Umber Dube<sup>2</sup>, Leon Bergen<sup>1</sup>
</b>
</p>
</div>

<p>
<sup>1</sup> Laboratory for Emerging Intelligence (LEI), University of California San Diego<br>
<sup>2</sup> Department of Dermatology, University of California San Diego<br>
<sup>3</sup> Elsevier<br>
<sup>*</sup> Equal contribution
</p>

<div align="center">

[![COLM 2026](https://img.shields.io/badge/COLM-2026-00629B?style=flat&labelColor=182B49)](https://colmweb.org/)&nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2604.16742-00629B?style=flat&labelColor=182B49)](https://arxiv.org/abs/2604.16742)
</div>

## Table of Contents
* [Overview](#overview)
* [How to evaluate the output](#how-to-evaluate-the-output)
  * [Prediction Format](#prediction-format)
  * [Example](#example)
  * [Evaluation Procedure](#evaluation-procedure)
  * [Evaluation Outputs](#evaluation-outputs)
  * [How to use it](#how-to-use-it)
* [Repository Goals](#repository-goals)
* [Repository Structure](#repository-structure)
* [Datasets](#datasets)
* [Benchmark Construction](#benchmark-construction)
  * [Decontamination pipeline](#1-decontamination-pipeline)
  * [Answer generation pipeline](#2-answer-generation-pipeline)
* [Evaluation Settings](#evaluation-settings)
* [RAG Prompt Data](#rag-prompt-data)
  * [Data Format](#data-format)
  * [Prompt Content](#prompt-content)
* [Challenge Schedule](#challenge-schedule)
* [Repository Contents](#repository-contents)
* [Intended Use](#intended-use)
* [Notes](#notes)
* [License](#license)
* [Citation](#citation)

Open-access benchmark and evaluation framework for **clinical trial outcome prediction**.

CT Open evaluates whether models can predict clinical trial outcomes **before results become publicly available**. The benchmark uses time-stamped train/test splits and a decontamination pipeline to reduce the risk that models rely on already-public trial results.

## Overview

Clinical trial outcome prediction is a high-stakes forecasting problem with consequences for patients, clinicians, pharmaceutical companies, and investors. CT Open focuses on prediction from publicly available trial information available before a fixed cutoff date. Trials with evidence of public results before the cutoff are excluded through an automated filtering and verification pipeline.

CT Open supports three question classes:

* **Superiority**: whether the treatment arm shows a statistically significant improvement over the comparator arm
* **Comparative Effect**: whether one arm is significantly better, worse, or not different from the comparator
* **Endpoint**: whether the endpoint is met, or whether at least one arm meets the endpoint


## How to evaluate the output

The `evaluations/evaluation.py` script provides the official evaluation logic for CT Open model outputs.

The main evaluation function is:

```python
evaluate_multiple_runs(list_of_eval_rets, benchmark_data)
````

It takes two main inputs:

* `benchmark_data`: the benchmark split that the model answered
* `list_of_eval_rets`: a list of result dictionaries, where each dictionary contains one run of model predictions

Each result dictionary maps a benchmark question to the model output for that question.

### Prediction Format

Each prediction should be stored in a dictionary.

The key is based on the benchmark key:

```python
(nctid, outcome_measure)
```

Since each benchmark item may contain multiple questions in `question_list_of_lists`, the prediction key must also include the question index:

```python
(nctid, outcome_measure, question_list_idx)
```

For example:

```python
(
    "NCT05799287",
    "{'measure': 'Change from baseline in urine protein creatinine ratio (UPCR)', 'description': 'Based on a 24-hour urine collections.', 'timeFrame': '39 weeks', 'outcome_type': 'primaryOutcomes'}",
    0
)
```

The value should be a string containing the model's final decision and optional reasoning:

```text
DECISION: (0.3, 0.7)

REASON: Telitacicept works by modulating B-cell activity, aiming to reduce the production of...
```

The `DECISION` field should contain a tuple of confidence scores. Each score corresponds to one answer option in the question. For a two-option question, the tuple should contain two values. For a three-option question, the tuple should contain three values.

The scores do not need to be normalized probabilities. For example, the following two outputs are treated equivalently after normalization:

```text
DECISION: (0.3, 0.7)
```

```text
DECISION: (3, 7)
```

The `REASON` field is not used for scoring. It is included to make model outputs easier to inspect, verify, and debug.

### Example

```python
eval_ret = {
    (
        "NCT05799287",
        "{'measure': 'Change from baseline in urine protein creatinine ratio (UPCR)', 'description': 'Based on a 24-hour urine collections.', 'timeFrame': '39 weeks', 'outcome_type': 'primaryOutcomes'}",
        0
    ): """DECISION: (0.3, 0.7)

REASON: Telitacicept works by modulating B-cell activity, aiming to reduce the production of..."""
}
```

If there is only one evaluation run, wrap the result dictionary in a list:

```python
list_of_eval_rets = [eval_ret]
```

Then run evaluation with:

```python
results = evaluate_multiple_runs(
    list_of_eval_rets=list_of_eval_rets,
    benchmark_data=benchmark_data
)
```

### Evaluation Procedure

For each model output, `evaluation.py`:

* locates the corresponding benchmark question using `(nctid, outcome_measure, question_list_idx)`
* retrieves the correct answer from `benchmark_data`
* extracts the confidence tuple after `DECISION:`
* normalizes the confidence scores into probabilities
* selects the answer option with the highest probability as the model prediction
* compares the prediction with the benchmark answer
* computes evaluation metrics across question categories

### Evaluation Outputs

The evaluation script reports results for the following categories:

* `Endpoint`
* `Superiority`
* `ComparativeEffect`

For each category, the script computes:

* **Weighted accuracy**: accuracy balanced across answer classes
* **Macro-F1**: F1 score averaged across classes
* **Cross-entropy**: how much probability the model assigns to the correct answer

The script also reports overall metrics by averaging across categories.

In general, stronger models should achieve higher weighted accuracy, higher macro-F1, and lower cross-entropy.

### How to use it:
Set the main function as follow:

```python
with open(PATH_TO_BENCHMARK_DATA, 'rb') as f:
    benchmark_data = pickle.load(f)

with open(PATH_TO_RESULT_1, 'rb') as f:
    ret_1 = pickle.load(f)

with open(PATH_TO_RESULT_2, 'rb') as f:
    ret_2 = pickle.load(f)

list_of_eval_rets = [ret_1, ret_2]
evaluate_multiple_runs(list_of_eval_rets, benchmark_data)
```

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

## RAG Prompt Data

The prompts used for our RAG method can be found at Evaluatios folder. They are divided into several parts since the github limit the file that bigger than 25MB.
Please merge them before use them.

The file stores the complete RAG prompts as a Python dictionary.

### Data Format

The RAG prompt file is organized as a dictionary. Each key is a tuple with three elements:

```python
(nctid, outcome_measure, question_index_in_each_nctid)
```

where:

* `nctid` is a string representing the ClinicalTrials.gov trial identifier.
* `outcome_measure` is a stringified dictionary describing the corresponding outcome measure.
* `question_index_in_each_nctid` is an integer indicating the question index within the same NCT ID.

This key format is the same as the key format used in the benchmark data. Therefore, users can directly use the key to locate the corresponding question in the benchmark.

### Prompt Content

Each value in the dictionary is a string containing the full RAG prompt for the corresponding question. The prompt can be divided into three main parts.

#### 1. Instruction

The first part provides the task instruction to the language model. It explains that the model will be given information about the current trial, together with information from historical trials that are considered relevant to the current trial. The model is then asked to answer the question and provide the reasoning behind its decision.

#### 2. Current Trial Information

The second part contains information about the current trial. This may include the biological mechanism of the drug, patient eligibility criteria, study arm information, and the specific proposed outcome measure.

#### 3. Relevant Historical Trial Information

The third part contains historical trials that are considered relevant to the current trial. Each retrieved trial is labeled as:

```text
Relevant Trial 0
Relevant Trial 1
Relevant Trial 2
...
```

Each relevant trial follows a structure similar to the current trial information, allowing the model to compare the target trial with related completed trials and reason about the likely outcome.



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

## License

Code in this repository is released under the Apache License 2.0 (see `LICENSE`).
The CT Open datasets (`datasets/` and the evaluation pickles) are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Citation

If you use this repository or build on the benchmark design, please cite the CT Open paper.

```bibtex
@inproceedings{
    wang2026ct,
    title={{CT} Open: An Open-Access, Uncontaminated, Live Platform for the Open Challenge of Clinical Trial Outcome Prediction},
    author={Jianyou Wang and Youze Zheng and Longtian Bao and Hanyuan Zhang and Qirui Zheng and Yuhan Chen and Yang Zhang and Matthew Feng and Maxim Khan and Aditya K Sehgal and Christopher D. Rosin and Mohan Paturi and Umber Dube and Leon Bergen},
    booktitle={Third Conference on Language Modeling},
    year={2026},
    url={https://openreview.net/forum?id=pbGCyTrXsl}
}
```
