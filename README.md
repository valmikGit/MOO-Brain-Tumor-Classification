# MOO-Brain-Tumor-Classification
Multi objective optimization for brain tumor classification

This repository implements a **Multi-Objective Optimization (MOO)**-based approach for **Brain Tumor Classification** using deep learning techniques. The goal is to classify brain tumor MRI scans into different tumor types efficiently while balancing multiple objectives such as accuracy, sensitivity, specificity, and model complexity.

## Project Highlights

- Leverages Multi-Objective Optimization to improve classification robustness.
- Evaluates models based on multiple criteria: accuracy, precision, recall, F1-score, and AUC.
- Applicable for medical image analysis and diagnostic support systems.

## Project Sructure
MOO-Brain-Tumor-Classification/
│
├── data/                         # Dataset directory (not included)
├── models/                       # Saved models and architecture files
├── notebooks/                    # Jupyter notebooks for experimentation
├── results/                      # Evaluation results, plots, and logs
├── utils/                        # Helper functions and scripts
├── requirements.txt              # Python dependencies
├── train.py                      # Script to train the model
├── evaluate.py                   # Script to evaluate model performance
└── README.md                     # This file

## Dataset

This project uses the **Brain MRI Images for Brain Tumor Detection** dataset, available from public sources like Kaggle. The dataset includes labeled MRI scans categorized into:

- **Glioma**
- **Meningioma**
- **Pituitary**
- **No Tumor**

Due to licensing, the dataset is not included in this repository. Please download it separately and place it in the `data/` directory.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/valmikGit/MOO-Brain-Tumor-Classification.git
   cd MOO-Brain-Tumor-Classification```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt```
4. Run the model
   ```bash
   python3 valmik_attemp7.py```
