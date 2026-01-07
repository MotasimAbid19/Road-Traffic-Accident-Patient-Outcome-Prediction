# Predictive Analysis of Patient Outcomes in Road Traffic Accidents (Ensemble ML)

Ensemble machine learning pipeline for predicting **patient outcome severity** after road traffic accidents using **emergency-response and hospital records (2020–2023)**. The project benchmarks baseline models and then applies ensemble methods with feature engineering and hyperparameter tuning for high-performing multi-class classification.

> Course project: **CSE445** (Final Phase).  


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
road-traffic-accident-patient-outcome-prediction/
├─ notebooks/
│  └─ Final_Phase_NoteBook.ipynb
├─ reports/
│  └─ FINAL_PHASE_REPORT.docx
│  └─ FINAL_PHASE_REPORT.pdf
├─ data/
│  ├─ .gitkeep
│  └─ RTA-DATA-2020-TO-JULY-2023.xlsx
│  └─ README.md
├─ assets/
│  └─ screenshots/
│     └─ .gitkeep
├─ .gitignore
├─ LICENSE
└─ README.md
```

## Dataset


- ~46k rows, 25 features  
- Target column: `PatientStatus`  
- Mix of numerical and categorical features, for example:  
  `Age`, `Gender`, `EducationTitle`, `EmergencyArea`, `Cause`, `Reason`,  
  `ResponseTime`, `InjuryType`, vehicle counts, etc.

### Local dataset placement

1. Put the dataset Excel file in `data/`, for example:
   - `data/RTA-DATA-2020-TO-JULY-2023.xlsx`
2. Update the notebook’s dataset-loading cell to something like:
```
import pandas as pd

df = pd.read_excel("data/RTA-DATA-2020-TO-JULY-2023.xlsx")
```
### Expected characteristics (from report)

- ~46k rows, 25 features  
- Target column: `PatientStatus`  
- Mix of numerical and categorical features, for example:  
  `Age`, `Gender`, `EducationTitle`, `EmergencyArea`, `Cause`, `Reason`,  
  `ResponseTime`, `InjuryType`, vehicle counts, etc.

### How to use your dataset locally

1. Put the dataset Excel file in `data/` (example):
   - `data/road_traffic_accidents.xlsx`
2. Update the notebook’s load cell to:
```data/road_traffic_accidents.xlsx```

>If your dataset filename or columns differ, update the notebook accordingly.

## Setup / Installation (Local)
```
python -m venv .venv
source .venv/bin/activate   # (Windows) .venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn xgboost matplotlib seaborn scipy jupyter
```

*Optional: create `requirements.txt` later for cleaner installs.*

---

## Usage

### Option A — Run in Google Colab 

1. Open `notebooks/CGS545_Final_Phase_Notebook.ipynb` in Colab
2. Upload the dataset file when prompted
3. Run all cells top-to-bottom

### Option B — Run locally

1. Place dataset in `data/`
2. Modify the dataset-loading cell (replace Colab upload with a local path)
3. Run the notebook with:

``` 
jupyter notebook

```


## Method Summary

This project predicts **patient outcome status** from road traffic accident records using an end-to-end ML workflow:

- **Data preparation:** load the dataset, handle missing values (notebook-defined strategy), and remove duplicates (if any).
- **Feature engineering:** construct a composite context feature to capture accident conditions:
  - `Cause_Reason = Cause + "_" + Reason + "_" + EmergencyArea`
- **Encoding:**
  - Encode categorical variables (mix of one-hot/ordinal/target-based approaches as implemented in the notebook).
  - Encode the target label `PatientStatus` into numeric classes.
- **Model training (benchmarking):**
  - Baselines: Logistic Regression, Decision Tree, SVM
  - Ensembles: Bagging, AdaBoost, Gradient Boosting, XGBoost, Voting, Stacking
- **Hyperparameter tuning:** use randomized hyperparameter search to find best settings for each model family.
- **Evaluation:** compare models after tuning and report final post-tuning performance.

---

## Results

### Post-Tuning Accuracy (Leaderboard)

| Rank | Model               | Post-Accuracy | Accuracy (%) |
|-----:|---------------------|--------------:|-------------:|
| 1    | GradientBoost       | 0.993609      | 99.3609%     |
| 2    | Stacking            | 0.993501      | 99.3501%     |
| 3    | Voting              | 0.993393      | 99.3393%     |
| 4    | XGBoost             | 0.993393      | 99.3393%     |
| 5    | Bagging             | 0.993176      | 99.3176%     |
| 6    | AdaBoost            | 0.993068      | 99.3068%     |
| 7    | Logistic Regression | 0.992959      | 99.2959%     |
| 8    | Decision Tree       | 0.992851      | 99.2851%     |

### Best Parameters (Post-Tuning)

> Some parameter dictionaries are truncated in the notebook display; keep them as-is unless you export full configs.

| Model               | Best Parameters |
|--------------------|-----------------|
| Logistic Regression | `{'C': 1}` |
| Decision Tree       | `{'max_depth': 10, 'min_samples_leaf': 5, 'min_...}` |
| SVM                 | `{'C': 1, 'gamma': 'scale', 'kernel': 'linear'}` |
| Bagging             | `{'max_features': 0.5779972601681014, 'max_samp...}` |
| AdaBoost            | `{'learning_rate': 0.3845401188473625, 'n_estim...}` |
| GradientBoost       | `{'learning_rate': 0.05666566321361543, 'max_de...}` |
| XGBoost             | `{'colsample_bytree': 0.855670976374325, 'learn...}` |
| Stacking            | `{'estimators': ['lr', 'dt', 'svm'], 'final_est...}` |
| Voting              | `{'voting': 'soft'}` |

### Accuracy: Pre vs Post-Tuning (Figure)

![Accuracy: Pre vs Post-Tuning](assets/screenshots/accuracy-pre-vs-post.png)


## Limitations

- Dataset is not shipped; reproduction requires access to the Excel file.
- Results depend on the split strategy and random seed used in the notebook (document/lock these for strict reproducibility).
- This is an academic project and **not** a clinical decision support tool.

---

## Roadmap

- [ ] Add `requirements.txt`
- [ ] Add fixed random seeds and a documented split strategy
- [ ] Add full parameter export (no truncation)
- [ ] Add an inference example (save/load best model)

---

## License

MIT
---

## Acknowledgements

- CSE445 — course context  
- scikit-learn and XGBoost libraries used for modeling

---

## Contact

- Motasim Abid — motasimabid19@gmail.com  
- Naima Zaman Roshni — naima.zaman@northsouth.edu

