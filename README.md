# ML-Home-Credit-Group-Project


# 🏠 Home Credit Default Risk Prediction

This repository contains my solution to the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) challenge hosted on Kaggle. The goal of this competition was to **predict the probability that a client will default on a loan**, using diverse financial and behavioral datasets.

I implemented a solution using **LightGBM**, achieving a ROC AUC score of **0.76108**, while the top of the public leaderboard was approximately **0.80**.

---

## 📌 Problem Statement

> Many individuals lack sufficient or documented credit history, making it hard to access loans from traditional institutions. Home Credit aims to support financial inclusion by leveraging alternative data sources to assess repayment capabilities responsibly.

Participants were provided with client data across **multiple tables** (e.g., credit history, previous applications, payment installments, etc.) and asked to predict the binary outcome of whether a client would default.

---

## 📁 Dataset Overview

The data consisted of several interlinked files:

* **`application_train.csv` / `application_test.csv`**: Main datasets with applicant and loan info.
* **`bureau.csv` & `bureau_balance.csv`**: Credit history from other institutions.
* **`previous_application.csv`**: All previous Home Credit loan applications.
* **`POS_CASH_balance.csv`**: Monthly POS/cash loan data.
* **`installments_payments.csv`**: Historical payment info.
* **`credit_card_balance.csv`**: Monthly balances on credit cards.
* **`HomeCredit_columns_description.csv`**: Metadata on all features.

Data relationships are visualized in the [column relationship diagram](#).

---

## 🧠 Approach

### 📊 Data Preprocessing

* Handled missing values and datatype inspections.
* Used `ydata_profiling` for EDA along with usage of histograms, lineplots and heatmaps
* Encoded categorical variables using `LabelEncoder`.

### ⚙️ Feature Engineering

* Engineered summary statistics from:

  * **bureau & bureau\_balance**
  * **POS\_CASH\_balance**
  * **credit\_card\_balance**
  * **installments\_payments**
  * **previous\_application**
* Merged all engineered features with the main application datasets.

### ⚡ Model Training

* Used `LightGBM` with:

  * Stratified Train/Test Split (80/20)
  * Custom parameter tuning via `GridSearchCV`
  * Early stopping for optimized training
* Trained final model on full training set using best-found parameters.

### 📈 Evaluation

* ROC AUC on validation set: **\~0.76**
* Final submission prepared by predicting probabilities on test set and saving to `submission.csv`.

---

## 🛠 Technologies Used

* Python (Pandas, NumPy, Scikit-learn)
* LightGBM
* ydata-profiling
* Matplotlib & Seaborn (for visualization)

---

## 🗂 File Structure

```text
├── data/                       # CSV files from Kaggle
├── notebooks/                 # (Optional) Jupyter or Colab notebooks
├── src/                       # Python scripts (if modularized)
├── submission.csv             # Final prediction file for Kaggle
├── HomeCredit_EDA.html        # ydata-profiling EDA report
├── README.md                  # Project documentation
└── home_credit_lightgbm.py    # Full training and prediction pipeline
```

---

## 🚀 Results

| Metric       | Score   |
| ------------ | ------- |
| ROC AUC (CV) | \~0.76  |
| My score     | 0.76108 |
| Top Score    | \~0.80  |

---

## 📌 Notes

* All data preprocessing and modeling were done in **Google Colab**, with files accessed via Google Drive.
* Feature importance analysis showed that predictive power was spread across many engineered features, each contributing marginally.

---

## 📬 Acknowledgements

* Thanks to **Home Credit Group** and **Kaggle** for the challenge and dataset.
* The dataset and problem statement are part of a real-world financial inclusion effort.

---

## 🔗 Links

* [Kaggle Competition Page](https://www.kaggle.com/competitions/home-credit-default-risk)

