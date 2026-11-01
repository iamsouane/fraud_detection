import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, average_precision_score
import sys

# Import de notre fonction de preprocessing
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_preprocessing import get_preprocessor

def main():
    # 1. Chemins
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, '..', 'data', 'raw', 'creditcard.csv')
    
    # 2. Chargement
    print("Chargement des données (cela peut prendre quelques secondes)...")
    df = pd.read_csv(data_path)
    
    # 3. Séparation X / y
    X = df.drop(columns=['Class'])
    y = df['Class']
    
    # 4. Split (Stratifié car 0.17% de fraudes)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Création du Pipeline
    # On utilise le Random Forest qui a donné les meilleurs résultats
    model_pipeline = make_pipeline(
        get_preprocessor(),
        RandomForestClassifier(
            n_estimators=100,
            max_depth=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1 # Utilise tous les cœurs du processeur pour aller plus vite
        )
    )
    
    # 6. Entraînement (Peut prendre 1 à 2 minutes)
    print("Entraînement du Random Forest...")
    model_pipeline.fit(X_train, y_train)
    
    # 7. Évaluation
    print("Évaluation sur le Test...")
    y_pred = model_pipeline.predict(X_test)
    y_proba = model_pipeline.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_proba)
    
    print("\n=== RÉSULTATS (Test) ===")
    print(f"Accuracy : {acc:.4f} (Trompeur !)")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    
    # 8. Sauvegarde du modèle
    model_path = os.path.join(base_dir, '..', 'fraud_model.joblib')
    joblib.dump(model_pipeline, model_path)
    print(f"\nModèle sauvegardé sous : {model_path}")

if __name__ == "__main__":
    main()