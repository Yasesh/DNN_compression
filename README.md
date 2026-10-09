# Heterogeneous Neural Network Compression via Adaptive SVD

A modular, mathematically rigorous framework for compressing deep learning architectures (Transformers and CNNs) through layer-wise rank allocation and singular value decomposition, designed to eliminate parameter bloat while preserving model intelligence.

---

## Project Overview

Modern deep learning models achieve high intelligence at the cost of massive parameter footprints. This project implements a **deterministic, energy-based adaptive Singular Value Decomposition (SVD)** framework to compress model layers heterogeneously based on their internal weight distributions. 

Instead of relying on a uniform compression ratio across all layers (which risks bottlenecking critical representations) or unstable reinforcement learning heuristics, our system calculates the optimal rank per layer on the fly using cumulative singular value energy thresholds.

---

## Key Features

* **Adaptive Energy-Based Rank Selection:** Computes the minimum rank required per layer to retain a target percentage of weight matrix energy (e.g., 90%), allocating higher ranks to critical layers and aggressive compression to redundant ones.
* **Strict Size-Check Safeguard (`SVDLinear`):** Mathematically verifies factorized parameter counts *before* replacing any layer. If a layer's decomposition risks parameter inflation, the wrapper automatically bypasses the split and retains the original layer.
* **Zero Training Overhead:** Operates purely through linear algebra matrix factorization, executing in seconds without requiring multi-episode training loops.
* **Modular Pipeline:** Cleanly separated into wrapper utilities (`src/svd_wrapper.py`), adaptive calculation engines (`src/adaptive_compressor.py`), and execution runners (`run_adaptive.py`).

---

## Models & Datasets

* **Target Architecture:** `bert-base-uncased` (109.4M parameters across 12 encoder blocks, comprising attention queries, keys, values, output projections, and FFN dense layers).
* **Evaluation / Calibration Data:** Standard pre-trained transformer weights from Hugging Face (`transformers` library).

---

## Project Structure

```text
DNN_compression/
├── README.md                    # Project documentation and benchmark logs
├── requirements.txt             # Core dependencies (torch, transformers, numpy, etc.)
├── svd_bert_benchmarks.csv      # Logged experimental benchmarks and metrics
├── run_adaptive.py              # Execution script for running adaptive SVD on BERT
└── src/
    ├── adaptive_compressor.py   # Energy-based cumulative variance and rank allocation logic
    ├── sensitivity_profiler.py  # Layer sensitivity and profiling utilities
    └── svd_wrapper.py           # SVDLinear factorization wrapper with size-check safeguards