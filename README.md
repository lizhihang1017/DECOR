# Adaptive Context Compression for Enhancing Retrieval-Augmented Generation

This repository contains the official implementation of the paper **"Adaptive Context Compression for Enhancing Retrieval-Augmented Generation"**. Our proposed DECOR adaptive compresses retrieved contexts to improve the efficiency and effectiveness of Retrieval-Augmented Generation (RAG) systems.

## 📖 Abstract

Context compression is pivotal for enabling Large Language Models (LLMs) to efficiently process long contexts in Retrieval-Augmented Generation (RAG) systems. However, existing approaches struggle with multi-hop reasoning: abstractive methods risk disrupting reasoning chains, extractive methods often lack adaptability, and many frameworks incur prohibitive computational costs. To address these challenges, we propose DECOR, a Coarse-to-Fine and Demand-Aware Adaptive Compression framework. DECOR operates via three core components: (1) Knowledge Structure Decomposition partitions documents into sentence-level units to ensure processing speed; (2) a Fine-Grained Knowledge Filter (FGKF) efficiently eliminates redundancy. Crucially, the FGKF is trained using atomic propositions as supervision signals, allowing it to inherit fine-grained semantic sensitivity while operating at the sentence level for low-latency inference; and (3) a Demand-Aware Knowledge Extractor explicitly models question-specific demands to preserve complete reasoning chains. We further employ a teacher-student distillation mechanism to ensure scalability. Extensive evaluations on five benchmarks demonstrate that DECOR significantly outperforms state-of-the-art methods, achieving a superior balance between accuracy and efficiency.

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-compatible GPU (recommended for training)

### Install dependencies
- pip install -r requirements.txt

## ⚡ DECOR inference
- bash run_filter.sh
- bash run_extractor.sh

## 📈 Training data acquisition
We're refactoring that part of the code! Please wait!

