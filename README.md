# Multi-Objective Neural Architecture Search with Pareto Frontier Analysis

## Overview
This framework performs neural architecture search with multi-objective optimization using Optuna, focusing on three key metrics: precision, recall, and class distribution bias. The system explores different weight combinations for these objectives and identifies Pareto-optimal solutions.

## Table of Contents
1. [Dataset Preparation](#dataset)
2. [Model Architecture](#architecture)
3. [Mathematical Formulation](#math)
4. [Hyperparameter Optimization](#optimization)
5. [Metrics Calculation](#metrics)
6. [Pareto Frontier Analysis](#pareto)
7. [Usage Instructions](#usage)
8. [Results Interpretation](#results)

## Dataset Preparation
- **Directory Structure**:
```bash
2/
├── Training/
│   ├── class0/
│   │   ├── image001.jpg
│   │   ├── image002.jpg
│   │   └── ...
│   └── class1/
│       ├── image101.jpg
│       ├── image102.jpg
│       └── ...
└── Testing/
    ├── class0/
    │   ├── test001.jpg
    │   └── ...
    └── class1/
        ├── test101.jpg
        └── ...
```

- **Transforms**:
```python
transforms.Resize((128, 128))
transforms.ToTensor()
```

- **Data Loading**:
  - 80-20 train-validation split
  - Batch size: 16
  - Automatic CUDA acceleration
