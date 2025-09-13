import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# ============ 1) Cargar datos ============
  
df = pd.read_csv("C:/Users/USUARIO/Downloads/Trabajo Samuel/titanic/train.csv") # <-- ajusta la ruta de train.csv

# ============ 2) Preprocesamiento ============
# Quitamos columnas con baja utilidad o muchos nulos
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

# Imputación simple
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# One-Hot Encoding para categóricas
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Separar X / y
X = df.drop('Survived', axis=1)
y = df['Survived']

# ============ 3) Split (sin fuga de información) ============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Escalado (beneficia a la Red Neuronal; a RF/GB no les afecta)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ============ 4) Modelos ============
# 4.1 Random Forest (clásico)
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# 4.2 Red Neuronal (MLPClassifier)
mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=1e-3,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1
)

# 4.3 Gradient Boosting (árboles aditivos)
gb = GradientBoostingClassifier(
    n_estimators=200,       # más árboles que el default
    learning_rate=0.05,     # tasa de aprendizaje más pequeña
    max_depth=3,            # profundidad de los árboles base
    random_state=42
)

# ============ 5) Función de evaluación ============
def evaluate_model(name, model, Xtr, ytr, Xte, yte, print_details=True):
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)

    proba_supported = hasattr(model, "predict_proba")
    y_proba = model.predict_proba(Xte)[:, 1] if proba_supported else None

    acc  = accuracy_score(yte, y_pred)
    prec = precision_score(yte, y_pred, average='binary', zero_division=0)
    rec  = recall_score(yte, y_pred, average='binary',  zero_division=0)
    f1   = f1_score(yte, y_pred, average='binary',      zero_division=0)
    auc  = roc_auc_score(yte, y_proba) if y_proba is not None else np.nan
    cm   = confusion_matrix(yte, y_pred)

    if print_details:
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print("Reporte de Clasificación:")
        print(classification_report(yte, y_pred, zero_division=0))
        print("Matriz de Confusión:")
        print(cm)
        if not np.isnan(auc):
            print(f"ROC-AUC: {auc:.4f}")

    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "fitted_model": model
    }

# ============ 6) Entrenar y evaluar ============
rf_metrics  = evaluate_model("Random Forest", rf, X_train_s, y_train, X_test_s, y_test, print_details=True)
mlp_metrics = evaluate_model("Red Neuronal (MLP)", mlp, X_train_s, y_train, X_test_s, y_test, print_details=True)
gb_metrics  = evaluate_model("Gradient Boosting", gb, X_train_s, y_train, X_test_s, y_test, print_details=True)

# Importancias del RF y GB (interpretabilidad)
importances_rf = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_metrics["fitted_model"].feature_importances_
}).sort_values("Importance", ascending=False)

print("\nImportancia de características (Random Forest):")
print(importances_rf)

# GradientBoostingClassifier también expone importancias
importances_gb = pd.DataFrame({
    "Feature": X.columns,
    "Importance": gb_metrics["fitted_model"].feature_importances_
}).sort_values("Importance", ascending=False)

print("\nImportancia de características (Gradient Boosting):")
print(importances_gb)

# ============ 7) Comparación lado a lado ============
comparison = pd.DataFrame([rf_metrics, mlp_metrics, gb_metrics])[
    ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
].set_index("model")

print("\n=== Comparación de modelos ===")
print(comparison.round(4))
# 1. Cargar el dataset

