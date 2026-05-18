# AEScorer: An Agentic Evidence-Grounded Framework for Graded Factuality Verification

This repository contains the official implementation and datasets for the anonymous paper:  
> **"AEScorer: An Agentic Evidence-Grounded Framework for Graded Factuality Verification"** (Anonymous ARR May submission).

---

## 🌟 Overview

Large Language Models (LLMs) often generate factually incorrect content, a phenomenon known as hallucination. Existing factuality verification methods primarily rely on binary judgments (e.g., correct or incorrect), which fail to distinguish the severity of errors. 

**AEScorer** addresses this limitation by framing factuality verification as a **graded scoring** problem rather than a purely binary decision. It is an **agentic evidence-grounded framework** that combines targeted evidence acquisition with calibrated scalar scoring.

---

## 🛠️ Framework Architecture

AEScorer operates through a two-stage pipeline:

1.  **Agentic Evidence Acquisition**: The model performs agentic search to gather verification-oriented evidence from external tools such as `WebSearch` and `LocalSearch`, following an iterative Think-Search-Observe loop.
2.  **Graded Scoring**: The acquired evidence is refined into compact evidence statements and verdict reasoning, then passed to a scoring module that predicts a continuous factuality score.

---

## 📂 Repository Structure

```text
.
├── ablation/               # Scripts for ablation studies 
├── corpora/                # Trusted document repositories (e.g., Wikipedia dumps)
├── datasets/               # Training triplets and GradedVeriBench data
├── scripts/                # Entry points for training and evaluation
├── src/                    # Core implementation of AEScorer modules
├── train/                  # Discriminative training logic and margin loss
├── requirements.txt        # Project dependencies
└── README.md
```

## 💡 Methodology Highlights

* **Graded Factuality Verification**: AEScorer models factuality as a continuous spectrum, enabling nuanced judgments within a single atomic fact or short response.
* **Agentic Evidence Grounding**: AEScorer autonomously retrieves external knowledge via `WebSearch` and `LocalSearch` when internal knowledge is insufficient for verification.
* **Evidence Refinement**: The search trajectory is condensed into useful evidence statements and verdict reasoning, providing high-signal input for the scoring stage.
* **Pairwise Preference Optimization**: The scoring module is trained with a margin ranking loss on pairwise preferences, avoiding the need for absolute scalar annotations.
* **LoRA Integration**: To reduce storage overhead, AEScorer uses Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

---

## 🚀 Execution Guide

To reproduce the results for **AEScorer**, follow the script execution order below:

### 1. Data Preparation
Construct the necessary training triplets and test samples for graded factuality verification:
* `bash create_data_train.sh`: Builds pairwise preference data for the graded scoring module.
* `bash create_data_test.sh`: Prepares the evaluation samples for the **GradedVeriBench** benchmark.

### 2. Model Training
Train the graded scoring module:
* `bash train_discriminative_module.sh`: Optimizes the verifier using a margin ranking loss and Low-Rank Adaptation (LoRA) for parameter efficiency.

### 3. AEScorer Inference Pipeline
Execute the full agentic evidence-grounded verification process:
* `bash infer_gen_and_cls.sh`: Runs the integrated AEScorer pipeline, including agentic evidence acquisition and graded scoring.
* `bash infer_gen_and_cls_detach.sh`: Runs the pipeline with the response detached into sub-claims first.
* `bash retrieval_launch.sh`: Launches the agentic search mechanism to retrieve external evidence from Google Search or Wikipedia.

### 4. Baseline Comparisons
Run the evaluation for competing generative and discriminative architectures:
* `bash infer_generative_ranking.sh`: Performs ranking using standard generative verifiers.
* `bash infer_generative_ranking_gpt4.sh`: Uses GPT-4 as the reference generative LLM-Judge.
* `bash infer_generative_scoring.sh`: Performs score-based evaluation using standard generative verifiers.
* `bash run_factscore.sh`: Executes the FactScore baseline for atomic claim verification.
* `bash run_minicheck.sh`: Executes the MiniCheck baseline for grounding-based fact-checking.
