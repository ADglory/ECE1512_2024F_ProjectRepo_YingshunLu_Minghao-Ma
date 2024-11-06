# Project A
# ECE1512 Project A: Dataset Distillation with Attention and Distribution Matching

This repository contains the code and documentation for **Project A** of **ECE1512 (Fall 2024)** by Yingshun Lu and Minghao Ma. The project focuses on dataset distillation techniques, specifically **Attention Matching** and **Distribution Matching**, to generate small, synthetic datasets that retain key characteristics of the original datasets, allowing efficient model training in resource-constrained environments.

## Project Overview

Dataset distillation is an emerging technique aimed at condensing large datasets into smaller, representative versions while preserving their core information. This project investigates the use of Attention Matching and Distribution Matching methods on two datasets: **MNIST** (handwritten digits) and **MHIST** (medical histopathology images). These distilled datasets enable deep learning models to be trained with minimal accuracy loss compared to training with the original datasets, making them ideal for applications in privacy-preserving machine learning, edge computing, and mobile devices.

### Objectives

- Apply **Attention Matching** to MNIST and MHIST datasets to generate distilled datasets that preserve attention maps.
- Implement **Distribution Matching** to align synthetic data distribution with real data distribution, ensuring high fidelity in synthetic datasets.
- Compare the efficacy of both methods in terms of classification accuracy, visual quality of synthetic data, computational efficiency, and generalization across architectures.

## Methods

### 1. Attention Matching
In Attention Matching, the objective is to align the attention maps of the real and synthetic data. This technique aims to preserve the regions in images that models consider important for classification, ensuring critical features are retained in the distilled dataset.

### 2. Distribution Matching
Distribution Matching is based on Maximum Mean Discrepancy (MMD), which seeks to minimize the distribution difference between real and synthetic data. This method often results in higher classification accuracy and better generalization on test data by closely aligning the synthetic dataset with the real data distribution.

## Results Summary

- **Attention Matching**: Achieved moderate test accuracy (50-65%) on the MNIST and MHIST datasets. The synthetic images generated show identifiable class features, but with some noise.
- **Distribution Matching**: Outperformed Attention Matching, reaching up to 97.3% accuracy on MNIST, closely approximating the performance of models trained on real data. This method also produced visually consistent synthetic images with clearer intra-class distinctions.

## Repository Structure

- **data/**: Contains the original MNIST and MHIST datasets.
- **code/**: Python scripts for dataset distillation and model training.
  - `attention_matching.py`: Implementation of Attention Matching.
  - `distribution_matching.py`: Implementation of Distribution Matching.
  - `utils.py`: Helper functions and utility scripts.
- **results/**: Contains results for both methods, including synthetic dataset samples and evaluation metrics.
- **notebooks/**: Jupyter notebooks for experimentation, visualization, and analysis.
  - `Task1_AttentionMatching.ipynb`: Notebook for Task 1, implementing Attention Matching.
  - `Task2_DistributionMatching.ipynb`: Notebook for Task 2, implementing Distribution Matching.
- **README.md**: Project documentation.

## Usage

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

## Running Experiments
- Download Datasets: Ensure that MNIST and MHIST datasets are downloaded and stored in the data/ directory.
