# Fine-Tuning Phi-4-Mini-Instruct for Medical Question Answering
### A Comparative Study of LoRA and QLoRA on MedMCQA

> **MSc Machine Learning Dissertation** — Liverpool John Moores University (LJMU / upGrad)  
> *Assessing Fine-Tuning Strategies for Medical Question Answering: A Comparative Study of LoRA and QLoRA*

---

## Overview

This repository contains the Google Colab training notebooks used in a dissertation-level experimental study comparing two parameter-efficient fine-tuning (PEFT) strategies — **LoRA** and **QLoRA** — applied to **Microsoft Phi-4-Mini-Instruct** (3.84B parameters) on the **MedMCQA** medical question-answering benchmark.

The study evaluates both strategies at two training scales (30K and 60K samples), measuring accuracy, F1 score, training loss, and GPU memory consumption.

---

## Repository Structure

```
├── Med_Finetune_LoRA_30K.ipynb          # LoRA fine-tuning — 30,000 training samples
├── Med_Finetune_LoRA_60K.ipynb          # LoRA fine-tuning — 60,000 training samples
└── Med_Finetune_Qlora_30K_and_60K.ipynb # QLoRA fine-tuning — 30K and 60K (single notebook)
```

---

## Experimental Setup

| Component | Detail |
|---|---|
| **Base Model** | `microsoft/Phi-4-mini-instruct` (3.84B parameters) |
| **Dataset** | `openlifescienceai/medmcqa` |
| **Task** | 4-way multiple-choice medical QA (A / B / C / D) |
| **Training platform** | Google Colab (A100 / T4 GPU) |
| **Adapter rank** | r = 16, alpha = 32 |
| **Target modules** | q_proj, k_proj, v_proj, o_proj, gate\_proj, up\_proj, down\_proj |
| **Dropout** | 0.05 |
| **Epochs** | 1 |
| **Effective batch size** | 16 (batch 2 × grad accumulation 8) |
| **Seed** | 42 |

### Key Differences Between Methods

| | LoRA | QLoRA |
|---|---|---|
| **Base model precision** | bfloat16 / float16 (full) | 4-bit NF4 (quantized) |
| **Quantization** | None — `bitsandbytes` uninstalled | `BitsAndBytesConfig` (double quant) |
| **VRAM requirement** | ~15.8 GB | ~7.2–7.4 GB |
| **`prepare_model_for_kbit_training`** | Not used | Required |
| **Merge capability** | `merge_and_unload()` supported | Not supported on 4-bit base |

---

## Results

| Run | Method | Training Samples | Accuracy | Macro F1 | Loss | Peak VRAM |
|---|---|---|---|---|---|---|
| Run 1 | LoRA | 30K | 52.35% | 51.82% | 1.52 | 15.8 GB |
| Run 2 | LoRA | 60K | 53.91% | 53.44% | 1.44 | 15.8 GB |
| Run 3 | QLoRA | 30K | 54.36% | 53.98% | 1.38 | 7.2 GB |
| Run 4 | QLoRA | 60K | **54.63%** | **54.18%** | **1.33** | 7.4 GB |

**Key finding:** QLoRA outperforms LoRA at both training scales while using approximately **54% less GPU memory**, making it the superior choice for resource-constrained fine-tuning of medical LLMs.

---

## Notebook Walkthrough

Each notebook follows the same structured pipeline:

1. **Install dependencies** — pinned library versions for reproducibility
2. **GPU check** — detects A100 vs T4 and adjusts precision accordingly
3. **Mount Google Drive** — saves adapters, logs, and metrics to Drive
4. **Load & preprocess MedMCQA** — filters invalid records, formats instruction-response prompts
5. **Load tokenizer** — sets pad token explicitly (Phi-4 Mini has none by default)
6. **Load base model** — bf16 full precision (LoRA) or 4-bit NF4 (QLoRA)
7. **Configure adapter** — identical LoRA hyperparameters across both methods for fair comparison
8. **Train with SFTTrainer** — using `processing_class=` (not deprecated `tokenizer=`)
9. **Save adapter weights & metadata** — to Google Drive
10. **Visualise training curves** — loss plots (train & validation) saved as PNG
11. **Evaluate** — logit-based inference on full validation set (no `generate()`)
12. **Subject-level accuracy analysis** — per-medical-subject breakdown
13. **Cross-method comparison table** — LoRA vs QLoRA side-by-side
14. **Optional: merge adapter** — LoRA only (`merge_and_unload()`)
15. **Release GPU memory** — explicit cleanup and runtime unassign

---

## Prompt Format

All runs use the same instruction-response prompt template:

```
You are a medical expert answering a multiple choice medical question.
Read the question carefully and select the single best option.
Reply with only one letter: A, B, C, or D.

Question: {question}

Options:
A) {opa}
B) {opb}
C) {opc}
D) {opd}

Answer: {A/B/C/D}
```

---

## Evaluation Method

Evaluation uses **logit-based next-token scoring** rather than `model.generate()`. The model scores each of the four candidate tokens (A, B, C, D) from the final prompt position logits and selects the highest. This approach is deterministic, fast, and avoids generation artefacts.

---

## Reproducibility Notes

- Prompt formatting is applied **before** dataset shuffle and sampling to ensure identical prompt distributions across runs
- Both LoRA and QLoRA use `dataset["train"].shuffle(seed=42).select(range(N))` for sampling — same sampling strategy for fair comparison
- `bitsandbytes` must be **fully uninstalled** for LoRA runs — it causes triton import crashes on some runtimes even when quantization is not used
- `SFTTrainer` requires `processing_class=tokenizer` (not the deprecated `tokenizer=` argument) with `trl >= 0.13.0`
- `transformers >= 4.48.0` is required for Phi-4-Mini-Instruct compatibility

### Pinned Library Versions

```
transformers==4.48.0
peft==0.14.0
trl==0.13.0
datasets==3.2.0
accelerate==1.2.1
```

---

## Hardware Requirements

| Method | Minimum GPU | Recommended |
|---|---|---|
| LoRA | T4 (16 GB) with batch size 1 | A100 (40 GB) |
| QLoRA | T4 (16 GB) comfortably | Any 8 GB+ GPU |

---

## Citation / Academic Context

This code supports the dissertation:

> *Assessing Fine-Tuning Strategies for Medical Question Answering: A Comparative Study of LoRA and QLoRA*  
> MSc Machine Learning — Liverpool John Moores University (LJMU / upGrad)  
> Supervisor: Dr. Rupal Bhargava

### References

- Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:2106.09685
- Dettmers et al. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* arXiv:2305.14314
- Pal et al. (2022). *MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering.* PMLR
- Microsoft (2024). *Phi-4-Mini-Instruct.* Hugging Face Hub

---

## License

This repository is shared for academic and research purposes. The MedMCQA dataset and Phi-4-Mini-Instruct model are subject to their respective licences from the original authors.
