# Adaptive Context Compression for Enhancing Retrieval-Augmented Generation

This repository contains the official implementation of the paper **"Adaptive Context Compression for Enhancing Retrieval-Augmented Generation"**. Our proposed DECOR adaptive compresses retrieved contexts to improve the efficiency and effectiveness of Retrieval-Augmented Generation (RAG) systems.

## 📖 Abstract

Context compression is crucial for enabling large language models to efficiently process long contexts in Retrieval-Augmented Generation (RAG) systems. Existing approaches, however, face limitations in multi-hop reasoning: abstractive methods may disrupt reasoning chains, extractive strategies often lack adaptive compression, and many frameworks incur high computational costs. To address these challenges, we propose DECOR, a Coarse-to-Fine and Demand-Aware Adaptive Compression framework. DECOR comprises three core components. The Knowledge Structure Decomposition partitions documents into fine-grained knowledge units, preserving semantically coherent reasoning blocks. The Fine-Grained Knowledge Filter evaluates relevance to remove redundant or distracting information, producing a concise, high-quality candidate set. The Demand-Aware Knowledge Extractor models question-specific demands to select essential evidence, ensuring complete reasoning chains for accurate multi-hop inference. We evaluate DECOR on five benchmark datasets, comparing it with state-of-the-art methods, as well as uncompressed, retrieval-based, and semantic chunking baselines. Experimental results demonstrate that DECOR outperforms existing methods, achieving a superior balance between accuracy and efficiency.

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

## 📊 Model training
