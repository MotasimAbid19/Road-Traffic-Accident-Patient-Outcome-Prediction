# Predictive Analysis of Patient Outcomes in Road Traffic Accidents (Ensemble ML)

Ensemble machine learning pipeline for predicting **patient outcome severity** after road traffic accidents using **emergency-response and hospital records (2020–2023)**. The project benchmarks baseline models and then applies ensemble methods with feature engineering and hyperparameter tuning for high-performing multi-class classification.

> Course project: **CSE445** (Final Phase).  
> Dataset is **not included** in this repository due to size and potential sensitivity.

---

## Overview

Road traffic accidents require rapid triage. This project predicts patient outcome classes (e.g., **Alive & stable**, **Alive & unstable/critical**, **Dead**) from accident context, patient demographics, injury information, and emergency response features.

Key points from the final report:
- Dataset: **46,190 records**, **25 features**, collected **2020–2023**
- Two-phase workflow: baseline models → ensemble models
- Feature engineering (notably a composite **Cause_Reason** feature) strongly improved performance
- Best reported ensemble: **Gradient Boosting ~99.36% accuracy** (see report for full details)

---

## Features

- End-to-end ML workflow in a single Jupyter notebook:
  - Data upload/loading
  - Missing-value handling and encoding
  - Feature engineering (**Cause_Reason** composite + encoding)
  - Train/validation/test split (stratified)
  - Baseline models (Logistic Regression, Decision Tree, SVM)
  - Ensemble models (Bagging, AdaBoost, Gradient Boosting, XGBoost, Voting, Stacking)
  - RandomizedSearchCV hyperparameter tuning for ensembles
  - Evaluation with accuracy/precision/recall/F1, confusion matrices, learning curves
- Final report included in `reports/` documenting methodology and results

---

## Tech Stack

- Python 3
- Jupyter Notebook / Google Colab
- Core libraries:
  - `pandas`, `numpy`
  - `scikit-learn`
  - `xgboost`
  - `matplotlib`, `seaborn`
  - `scipy`

---

## Repository Structure

```text
notebooks/   Jupyter notebook implementation (final pipeline)
reports/     Final project report (documentation + results)
data/        Dataset folder (not tracked in git)
assets/      Screenshots/figures for README (optional)
