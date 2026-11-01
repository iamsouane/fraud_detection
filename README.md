# Credit Card Fraud Detection - End-to-End ML Pipeline

## 📌 Project Overview
This project builds a Machine Learning pipeline to detect fraudulent credit card transactions. The main challenge is the **extreme class imbalance** (only 0.17% of transactions are frauds).

The objective is to build a robust model that maximizes **Recall** (catching frauds) while maintaining a good **F1-score** (avoiding too many false alarms).

## 🛠️ Tech Stack
- **Python 3.12**
- **Pandas & NumPy**
- **Scikit-Learn** (Pipelines, ColumnTransformer, RandomForest)
- **Joblib** (Model serialization)

## 🧠 Methodology
1. **Exploratory Data Analysis (EDA)**: Identified extreme class imbalance (99.83% normal vs 0.17% fraud).
2. **Data Leakage Prevention**: Used `train_test_split` with `stratify=y` to ensure frauds are present in both Train and Test.
3. **Feature Engineering**: 
   - The dataset contains PCA-transformed features (`V1` to `V28`) that are already scaled.
   - Only `Time` and `Amount` were scaled using `StandardScaler` via a `ColumnTransformer`.
4. **Handling Imbalance**: Used `class_weight='balanced'` to penalize errors on the minority class (frauds).
5. **Modeling**: Used a **Random Forest Classifier** (`max_depth=20`) which significantly outperformed Logistic Regression.
6. **Threshold Tuning**: Analyzed the Precision/Recall trade-off to find the optimal decision threshold (0.5).

## 📊 Results

After training and evaluating on the unseen test set:

```text
=== RÉSULTATS (Test) ===
Accuracy : 0.9996 (Misleading metric in this context)
Recall   : 0.8163 (82% of frauds caught)
F1-score : 0.8696 (Excellent balance)
PR-AUC   : 0.8635

Interpretation:

Accuracy (99.96%): Not meaningful here due to the imbalance.
Recall (81.6%): The model successfully detects 82% of real frauds.
F1-Score (86.9%): Proves the model is highly precise when it predicts a fraud, with very few false alarms.
PR-AUC (86.3%): The model is excellent at ranking frauds above normal transactions.
```

📂 Repository Structure

```text
fraud-detection/
│
├── data/
│   └── raw/                 # Raw CSV (ignored in git - 150MB file)
│
├── notebooks/
│   └── 01_fraud_eda.ipynb   # EDA and prototyping
│
├── src/
│   ├── data_preprocessing.py # ColumnTransformer logic
│   └── train.py              # Production training script
│
├── .gitignore
├── requirements.txt
└── README.md
```

🚀 How to Run
```text
1. Clone the repository

git clone https://github.com/TON_PSEUDO/fraud-detection.git
cd fraud-detection

2. Download the dataset

You must download creditcard.csv from Kaggle and place it in data/raw/.

3. Set up the environment

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

4. Run the training pipeline

python src/train.py
