# -*- coding: utf-8 -*-
"""
Algoritmo adicional: Gradient Boosting (clasificación o regresión)
Autor: Equipo
Uso: Edita CSV_PATH y TARGET abajo y ejecuta:  python extra_gradient_boosting.py
"""
import os, sys, json, math, warnings
import numpy as np
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report,
                             confusion_matrix, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt
import joblib
warnings.filterwarnings("ignore")

# ========= CONFIGURACIÓN =========
CSV_PATH = "ruta/a/tu_dataset.csv"             # <-- CAMBIA
TARGET   = "nombre_de_la_columna_objetivo"     # <-- CAMBIA
PROBLEM_TYPE = "auto"  # "auto" | "clasificacion" | "regresion"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = None  # GradientBoosting no usa n_jobs. (GridSearch sí paraleliza en CV)

# ========= UTILIDADES =========
def detectar_tipo_problema(y: pd.Series) -> str:
    """Detecta 'clasificacion' o 'regresion' según el target."""
    if y.dtype.kind in "biu":  # enteros
        # Si hay muy pocas clases únicas, asumimos clasificación
        n_uniq = y.nunique(dropna=True)
        if n_uniq <= max(20, int(0.05*len(y))):
            return "clasificacion"
        else:
            return "regresion"
    if y.dtype.kind in "f":
        return "regresion"
    # Objetos o categóricas => clasificación
    return "clasificacion"

def obtener_columnas(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]
    return num_cols, cat_cols

def construir_preprocesador(num_cols, cat_cols):
    num_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols)
        ]
    )
    return pre

def nombres_features(transformer: ColumnTransformer, num_cols, cat_cols) -> List[str]:
    """Recupera nombres de características luego del ColumnTransformer."""
    output_features = []
    # num_cols (después del scaler mantienen el nombre original)
    output_features.extend(num_cols)
    # cat_cols -> OneHotEncoder get_feature_names_out
    try:
        ohe = transformer.named_transformers_["cat"]["encoder"]
        cat_names = ohe.get_feature_names_out(cat_cols).tolist()
    except Exception:
        cat_names = []
    output_features.extend(cat_names)
    return output_features

def asegurar_salida():
    outdir = "salidas_gb"
    os.makedirs(outdir, exist_ok=True)
    return outdir

def plot_importancias(model, feat_names, outdir):
    if not hasattr(model, "feature_importances_"):
        return None
    importancias = model.feature_importances_
    # Top 20
    idx = np.argsort(importancias)[::-1][:20]
    plt.figure(figsize=(8, 6))
    plt.barh(range(len(idx)), importancias[idx][::-1])
    plt.yticks(range(len(idx)), [feat_names[i] for i in idx][::-1])
    plt.xlabel("Importancia")
    plt.title("Importancia de características (Top 20) - Gradient Boosting")
    path = os.path.join(outdir, "importancias_caracteristicas.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path

def plot_confusion(y_true, y_pred, labels, outdir):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title("Matriz de confusión")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    path = os.path.join(outdir, "matriz_confusion.png")
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path

def plot_roc_auc(y_true, y_proba, outdir):
    # Solo binaria
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend(loc="lower right")
    path = os.path.join(outdir, "roc_auc.png")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    return path, roc_auc

def main():
    outdir = asegurar_salida()

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] No se encontró el CSV en: {CSV_PATH}")
        print("Edita la variable CSV_PATH al inicio del script.")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    if TARGET not in df.columns:
        print(f"[ERROR] La columna objetivo '{TARGET}' no existe en el CSV.")
        sys.exit(1)

    y = df[TARGET]
    X = df.drop(columns=[TARGET])

    # Detectar tipo de problema si es auto
    problem = PROBLEM_TYPE
    if problem == "auto":
        problem = detectar_tipo_problema(y)

    # Columnas por tipo
    num_cols, cat_cols = obtener_columnas(X)

    # Preprocesador
    pre = construir_preprocesador(num_cols, cat_cols)

    # Modelo base y param grid
    if problem == "clasificacion":
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
        param_grid = {
            "model__n_estimators": [150, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0]
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    else:
        base_model = GradientBoostingRegressor(random_state=RANDOM_STATE)
        param_grid = {
            "model__n_estimators": [150, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4],
            "model__subsample": [0.7, 0.85, 1.0]
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pipe = Pipeline(steps=[("pre", pre), ("model", base_model)])

    # Split
    if problem == "clasificacion":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    # Grid Search (pequeño)
    gs = GridSearchCV(pipe, param_grid, cv=cv, n_jobs=N_JOBS, verbose=0)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    best_params = gs.best_params_
    # Entrenar final en train ya está hecho por GS; evaluamos
    y_pred = best.predict(X_test)

    # Obtener nombres de features post-transformación
    best_pre = best.named_steps["pre"]
    feat_names = nombres_features(best_pre, num_cols, cat_cols)

    # Métricas
    reporte_md = []
    reporte_md.append(f"# Resultados Gradient Boosting ({problem})\n")
    reporte_md.append(f"**Mejores parámetros (GridSearchCV):** `{best_params}`\n")

    if problem == "clasificacion":
        acc = accuracy_score(y_test, y_pred)
        f1m = f1_score(y_test, y_pred, average="macro")
        reporte_md.append(f"- Accuracy: **{acc:.4f}**")
        reporte_md.append(f"- F1 macro: **{f1m:.4f}**")

        # Probabilidades y ROC (si binaria)
        roc_path, roc_auc = None, None
        labels = sorted(y_test.unique().tolist())
        if len(labels) == 2:
            try:
                y_proba = best.predict_proba(X_test)[:, 1]
                roc_path, roc_auc = plot_roc_auc(y_test.map({labels[0]:0, labels[1]:1}), y_proba, outdir)
                reporte_md.append(f"- ROC AUC: **{roc_auc:.4f}**")
            except Exception:
                pass

        # Matriz de confusión
        cm_path = plot_confusion(y_test, y_pred, labels, outdir)
        # Reporte por clase
        cls_rep = classification_report(y_test, y_pred)
        reporte_md.append("\n### Classification report\n")
        reporte_md.append("```\n" + cls_rep + "\n```")

        # Importancias
        imp_path = plot_importancias(best.named_steps["model"], feat_names, outdir)

    else:  # regresión
        mae = mean_absolute_error(y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        reporte_md.append(f"- MAE: **{mae:.4f}**")
        reporte_md.append(f"- RMSE: **{rmse:.4f}**")
        reporte_md.append(f"- R²: **{r2:.4f}**")
        # Importancias
        imp_path = plot_importancias(best.named_steps["model"], feat_names, outdir)

    # Guardar reporte
    report_path = os.path.join(outdir, "reporte_resultados.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(reporte_md))

    # Guardar modelo
    joblib.dump(best, os.path.join(outdir, "gb_model.joblib"))

    print("[OK] Entrenamiento y evaluación listos.")
    print(f"- Reporte: {report_path}")
    print(f"- Importancias: {os.path.join(outdir, 'importancias_caracteristicas.png')}")
    if problem == "clasificacion":
        print(f"- Matriz de confusión: {os.path.join(outdir, 'matriz_confusion.png')}")
        print(f"- Curva ROC (si binaria): {os.path.join(outdir, 'roc_auc.png')}")

if __name__ == "__main__":
    main()
