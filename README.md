# Adaptive Context Compression for Enhancing Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper **"Adaptive Context Compression for Enhancing Retrieval-Augmented Generation"**. Our proposed DECOR model dynamically compresses retrieved contexts to improve the efficiency and effectiveness of Retrieval-Augmented Generation (RAG) systems.

## 📖 Abstract

Context compression is crucial for enabling large language models to efficiently process long contexts in Retrieval-Augmented Generation (RAG) systems. Existing approaches, however, face limitations in multi-hop reasoning: abstractive methods may disrupt reasoning chains, extractive strategies often lack adaptive compression, and many frameworks incur high computational costs. To address these challenges, we propose DECOR, a Coarse-to-Fine and Demand-Aware Adaptive Compression framework. DECOR comprises three core components. The Knowledge Structure Decomposition partitions documents into fine-grained knowledge units, preserving semantically coherent reasoning blocks. The Fine-Grained Knowledge Filter evaluates relevance to remove redundant or distracting information, producing a concise, high-quality candidate set. The Demand-Aware Knowledge Extractor models question-specific demands to select essential evidence, ensuring complete reasoning chains for accurate multi-hop inference. We evaluate DECOR on five benchmark datasets, comparing it with state-of-the-art methods, as well as uncompressed, retrieval-based, and semantic chunking baselines. Experimental results demonstrate that DECOR outperforms existing methods, achieving a superior balance between accuracy and efficiency.
