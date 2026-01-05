# DiVA: Fine-grained Factuality Verification with Agentic-Discriminative Verifier

This repository contains the official implementation and datasets for the anonymous paper:  
> **"DiVA: Fine-grained Factuality Verification with Agentic-Discriminative Verifier"** (ACL 2026 Anonymous Submission).

---

## 🌟 Overview

Large Language Models (LLMs) often generate factually incorrect content, a phenomenon known as hallucination. Existing factuality verification methods primarily rely on binary judgments (e.g., correct or incorrect), which fail to distinguish the severity of errors. 

**DiVA (Agentic-Discriminative Verifier)** is a hybrid framework that bridges this gap by synergizing the **agentic search capabilities** of generative models with the **precise scoring aptitude** of discriminative models.

---

## 🛠️ Framework Architecture

DiVA operates through a specialized three-stage pipeline:

1.  **Agentic Search**: A generative module utilizes reasoning and tool-use capabilities to autonomously retrieve external knowledge from sources like Google Search and Wikipedia. It follows a loop of Thought, Action, and Observation.
2.  **Context Compression**: The retrieved search trajectory is condensed to filter out irrelevant information while preserving key facts and reasoning steps.
3.  **Score Prediction**: The compressed context is fed into a discriminative module (equipped with a regression head) to output a continuous factuality score.

---

## 📂 Repository Structure

```text
.
├── ablation/               # Scripts for ablation studies 
├── corpora/                # Trusted document repositories (e.g., Wikipedia dumps)
├── datasets/               # Training triplets and FGVeriBench data
├── scripts/                # Entry points for training and evaluation
├── src/                    # Core implementation of DiVA modules
├── train/                  # Discriminative training logic and margin loss
├── requirements.txt        # Project dependencies
└── README.md

## 💡 Methodology Highlights

* **Agentic Reasoning**: DiVA empowers an LLM agent to autonomously retrieve external knowledge via `WebSearch` and `LocalSearch` when internal knowledge is insufficient.
* **Context Compression**: We leverage a generative module to condense the search trajectory, filtering out irrelevant information to provide high-signal input for the discriminator.
* **Pairwise Training**: The discriminative module is optimized using a margin ranking loss on pairwise data, eliminating the need for absolute score annotations.
* **LoRA Integration**: To minimize storage overhead, DiVA employs Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning.

---

## 🚀 Execution Guide

To reproduce the results for **DiVA**, follow the script execution order below:

### 1. Data Preparation
Construct the necessary training triplets and test samples for fine-grained verification:
* `bash scripts/create_data_train.sh`: Builds pairwise preference data (query, factual response, and non-factual response) for the discriminative module.
* `bash scripts/create_data_test.sh`: Prepares the evaluation samples for the **FGVeriBench** benchmark.

### 2. Model Training
Train the specialized discriminative module:
* `bash scripts/train_discriminative_module.sh`: Optimizes the verifier using a margin ranking loss and Low-Rank Adaptation (LoRA) for parameter efficiency.

### 3. DiVA Inference Pipeline
Execute the full agentic-discriminative verification process:
* `bash scripts/infer_gen_and_cls.sh`: Runs the integrated three-stage pipeline (Agentic Search, Context Compression, and Score Prediction).
* `bash scripts/infer_gen_and_cls_detach.sh`: Runs the integrated three-stage pipeline with the response detached to sub-claims first.
* `bash scripts/retrieval_launch.sh`: Specifically launches the agentic search mechanism to retrieve external evidence from Google Search or Wikipedia.

### 4. Baseline Comparisons
Run the evaluation for competing generative and discriminative architectures:
* `bash scripts/infer_generative_ranking.sh`: Performs ranking using standard generative verifiers.
* `bash scripts/infer_generative_ranking_gpt4.sh`: Uses GPT-4 as the reference generative LLM-Judge.
* `bash scripts/infer_generative_scoring.sh`: Performs scoring based standard generative verifiers.
* `bash scripts/run_factscore.sh`: Executes the FactScore baseline for atomic claim verification.
* `bash scripts/run_minicheck.sh`: Executes the MiniCheck baseline for grounding-based fact-checking.

---

## 🙏 Acknowledgments

This work was supported by JST K Program Grant Number JPMJKP24C3, Japan, and the National Natural Science Foundation of China (62276077).
