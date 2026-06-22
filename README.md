# Credit Card Fraud Detection

A machine learning pipeline for detecting fraudulent credit card transactions. Compares Logistic Regression and Random Forest classifiers across different preprocessing strategies (SMOTE oversampling, feature selection, and original data).

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — ULB Machine Learning Group via Kaggle.

The dataset contains transactions made by European cardholders in September 2013. It includes 284,807 transactions over two days, of which 492 (0.17%) are fraudulent. Features `V1`–`V28` are PCA-transformed for confidentiality; `Time`, `Amount`, and `Class` (0 = normal, 1 = fraud) are the remaining columns.

## Project Structure

```
.
├── creditcard.csv        # Dataset (download from Kaggle)
└── fraud_detection.py    # Main pipeline script
```

## Pipeline Overview

```
Load Data → Explore → Clean → Analyze → Feature Selection
    → Scale & Split → SMOTE → Train Models → Evaluate → Compare
```

### Steps

| Stage | What happens |
|---|---|
| **Exploration** | Shape, dtypes, null check, first rows |
| **Cleaning** | Duplicate removal |
| **Analysis** | Summary stats, class distribution, correlation heatmap |
| **Feature selection** | Keeps features with \|correlation\| > 0.15 with `Class` |
| **Scaling** | `StandardScaler` on all numeric features |
| **Train/test split** | 80/20, `random_state=1` |
| **SMOTE** | Oversamples minority (fraud) class in training data |
| **Training** | 6 models across 2 algorithms × 3 data variants |
| **Evaluation** | Classification report + confusion matrix per model |
| **Comparison** | Bar chart of Precision / Recall / F1 for the fraud class |

## Models Trained

| # | Model | Training Data |
|---|---|---|
| 1 | Logistic Regression | SMOTE-balanced |
| 2 | Logistic Regression | Important features only |
| 3 | Logistic Regression | Original (imbalanced) |
| 4 | Random Forest | SMOTE-balanced |
| 5 | Random Forest | Important features only |
| 6 | Random Forest | Original (imbalanced) |

## Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

## Usage

1. Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root.
2. Run the pipeline:

```bash
python fraud_detection.py
```

The script will print evaluation metrics to the console and display visualizations for class distribution, amount distribution by class, correlation heatmap, and a final model comparison bar chart.

## Key Functions

| Function | Description |
|---|---|
| `dataset_overview()` | Prints shape, dtypes, and null counts |
| `correlation_analysis()` | Heatmap of feature correlations with `Class` |
| `get_important_features()` | Filters features by correlation threshold |
| `apply_smote()` | Balances training data via oversampling |
| `train_logistic_regression()` | Fits a Logistic Regression model |
| `train_random_forest()` | Fits a Random Forest classifier |
| `evaluate_model()` | Prints classification report and confusion matrix |
| `predict_fraud()` | Predicts fraud status for a single transaction |
| `compare_models()` | Bar chart comparing all models on fraud-class metrics |

## Notes

- Evaluation focuses on the **fraud class** (class `1`) since it is the minority and the class of interest. Pay close attention to **recall** — a missed fraud is more costly than a false alarm.
- SMOTE is applied **only to training data** to prevent data leakage.
- The feature importance threshold (`0.15`) can be adjusted in `get_important_features()`.
