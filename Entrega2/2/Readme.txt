# Titanic – Random Forest (Aprendizaje Supervisado)

Titanic — Aprendizaje Supervisado (Random Forest vs MLP)

1) Dataset

Usamos el clásico dataset **Titanic** de Kaggle (CSV):

- Página: Titanic – Machine Learning from Disaster
- Archivos relevantes: `train.csv` (incluye la etiqueta `Survived`), `test.csv` (sin etiquetas, opcional)

**Cómo descargar:**
1. Crear cuenta en Kaggle (gratis).
2. Ir a la página del concurso y pulsar **Download All**, link "https://www.kaggle.com/c/titanic/data?utm_source".
3. Descomprimir y ubicar `train.csv` en tu carpeta del proyecto.

> En este repo se asume la ruta:
> ```
> C:\Users\USUARIO\OneDrive\Documentos\TrabajoIA\titanic\train.csv
> ```

2) Requisitos y entorno

- Python 3.10+ (probado en 3.13)
- Librerías:
  - `pandas` (manipulación de datos)
  - `scikit-learn` (modelado y métricas)
  - `numpy` (soporte numérico)

**Instalación (global o en venv):**
pip install scikit-learn pandas

3) Descripción del dataset

Fuente: Competencia “Titanic – Machine Learning from Disaster” (Kaggle).

Objetivo del problema: Clasificación binaria — predecir si un pasajero sobrevivió (Survived = 1) o no (Survived = 0) usando información del viaje y del pasajero.

Archivos:

train.csv: contiene la etiqueta Survived (usada para entrenar y evaluar).

test.csv: no se usa en este trabajo (no tiene etiqueta).

Número de registros y variables (train.csv): ~891 filas y las columnas comunes del diccionario de datos:
PassengerId, Survived (target), Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.

Variables efectivamente usadas por el modelo (tras preprocesamiento):
Pclass, Age, SibSp, Parch, Fare, Sex_(male), Embarked_(Q), Embarked_(S).

Ruta del CSV usada en el script (ajustable):
CSV_PATH = C:\Users\USUARIO\OneDrive\Documentos\TrabajoIA\titanic\train.csv

4) Preprocesamiento realizado

El mismo pipeline se aplica a ambos modelos para una comparación justa.

a) Limpieza de datos faltantes

Columnas eliminadas: PassengerId, Name, Ticket, Cabin (baja utilidad y/o demasiados nulos).

Imputaciones:

Age → mediana.

Embarked → moda (valor más frecuente).

b) Codificación de variables categóricas

One-Hot Encoding con pd.get_dummies(..., drop_first=True) para Sex y Embarked.

Se generan columnas binarias (por ejemplo, Sex_male, Embarked_Q, Embarked_S) evitando colinealidad perfecta.

c) Escalado / normalización

StandardScaler aplicado después del split (para evitar fuga de información):

X_train_s = scaler.fit_transform(X_train)

X_test_s = scaler.transform(X_test)

Justificación: MLP requiere datos estandarizados; Random Forest no lo necesita pero no se ve perjudicado por el escalado.

d) División en train/test

train_test_split(test_size=0.20, random_state=42, stratify=y)

Estratificación por y (mantiene la proporción de clases en train y test).

5) Entrenamiento de los dos modelos y parámetros

Ambos modelos se entrenan y evalúan con el mismo conjunto de datos preprocesado.

5.1 Random Forest (modelo clásico elegido)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


Ventajas: robusto, captura no linealidades, poco sensible a outliers/escala.

Interpretabilidad: feature_importances_.

5.2 Red Neuronal — MLPClassifier
from sklearn.neural_network import MLPClassifier
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


Requiere escalado.

early_stopping evita sobreajuste usando una fracción de validación interna.

6) Evaluación de resultados

Se usa la función evaluate_model(...) del script, que calcula y muestra en consola:

a) Métricas de rendimiento (por modelo)

Accuracy

Precision (para la clase positiva 1)

Recall (para la clase positiva 1)

F1-score

ROC-AUC (si el modelo expone predict_proba; ambos lo hacen)

Reporte de clasificación (precision/recall/F1 por clase y macro/weighted)

Matriz de confusión

Además, se imprime la importancia de características del Random Forest y una tabla comparativa lado a lado:

=== Random Forest ===
Accuracy: 0.81XX
Reporte de Clasificación:
...
Matriz de Confusión:
[[TN FP]
 [FN TP]]
ROC-AUC: 0.8XXX  (si aplica)

=== Red Neuronal (MLP) ===
Accuracy: 0.8XXX
Reporte de Clasificación:
...
Matriz de Confusión:
[[TN FP]
 [FN TP]]
ROC-AUC: 0.8XXX

Importancia de características (Random Forest):
Feature        Importance
Fare           ...
Sex_male       ...
Age            ...
...

=== Comparación de modelos ===
                      accuracy  precision  recall     f1  roc_auc
model
Random Forest           0.81xx     0.7xxx  0.7xxx  0.7xx   0.8xxx
Red Neuronal (MLP)      0.8xxx     0.7xxx  0.7xxx  0.7xx   0.8xxx
