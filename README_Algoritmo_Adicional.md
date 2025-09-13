# Algoritmo adicional: Gradient Boosting (Boosting por gradiente)

Este documento cumple el punto **“Un algoritmo adicional a investigar y aplicar (no visto en clase)”**.
La técnica elegida es **Gradient Boosting**, una familia de modelos de *boosting* que construye de forma secuencial
múltiples modelos débiles (típicamente árboles de decisión poco profundos), donde cada nuevo modelo corrige los errores
del conjunto anterior. El resultado es un **ensamble aditivo** que suele lograr alto desempeño con buen control del sesgo/varianza.

---

## 1) ¿Qué es Gradient Boosting? (Investigación)

- **Idea base:** entrenar modelos débiles de manera secuencial. Cada iteración entrena un árbol sobre los **residuos**
  (o gradiente del error) del ensamble acumulado hasta el momento.
- **Función de pérdida:** para clasificación se usan pérdidas como *deviance/log-loss*; para regresión, MSE/MAE.
- **Aprendizaje aditivo:** el modelo final es la suma ponderada de muchos árboles pequeños (*stumps* o con poca profundidad).
- **Regularización:** se controla con `n_estimators` (número de árboles), `learning_rate` (tamaño de paso) y `max_depth`
  (complejidad de cada árbol). Un `learning_rate` pequeño con más árboles suele generalizar mejor.

**Ventajas**
- Suele rendir **muy bien** en datos tabulares.
- Permite **interpretar** importancia de variables.
- Tolerante a *outliers* moderados y relaciones no lineales.

**Desventajas**
- Entrenamiento más **lento** que un árbol simple o regresión logística.
- Sensible a *overfitting* si no se regula bien.
- Requiere más **tuning** de hiperparámetros.

**Cuándo usarlo**
- Problemas tabulares de **clasificación o regresión** con mezcla de variables numéricas y categóricas (tras *one-hot*).
- Cuando Random Forest o un árbol simple quedan cortos, pero no quieres usar librerías externas (XGBoost/LightGBM).

---

## 2) Implementación (código y pasos)

Incluyo un **script listo** (`extra_gradient_boosting.py`) que:
1. Carga tu CSV de Kaggle.
2. Separa *features* y `target`.
3. Hace **preprocesamiento** automático (imputación, *one-hot*, escalado).
4. Detecta si el problema es **clasificación** o **regresión** (o puedes forzarlo).
5. Entrena y hace **búsqueda de hiperparámetros** ligera.
6. Genera **métricas** y figuras (matriz de confusión, curva ROC si aplica, importancias).
7. Guarda un **reporte** en Markdown con resultados y parámetros óptimos.

### Requisitos
- Python 3.9+
- `pip install -r requirements.txt`

### Uso
Edita dentro del script las dos variables al inicio:
```python
CSV_PATH = "ruta/a/tu_dataset.csv"   # <- CAMBIA ESTO
TARGET   = "nombre_de_la_columna_objetivo"  # <- CAMBIA ESTO
PROBLEM_TYPE = "auto"  # "auto" | "clasificacion" | "regresion"
TEST_SIZE = 0.2
RANDOM_STATE = 42
```

Luego ejecuta:
```bash
python extra_gradient_boosting.py
```

El script producirá en la carpeta `./salidas_gb/`:
- `reporte_resultados.md` (métricas, mejores parámetros y resumen)
- `importancias_caracteristicas.png`
- `matriz_confusion.png` (solo clasificación)
- `roc_auc.png` (solo clasificación binaria)
- `gb_model.joblib` (modelo entrenado)

---

## 3) Parámetros y búsqueda (qué afinamos)
- `n_estimators`: número de árboles en el ensamble (ej: 100–600).
- `learning_rate`: tamaño de paso (ej: 0.01–0.2).
- `max_depth`: profundidad de cada árbol (ej: 2–5).
- `subsample`: fracción de muestras por árbol (ej: 0.6–1.0).

La búsqueda incluida es **ligera** (para tiempos razonables). Puedes ampliarla si tu máquina aguanta.

---

## 4) Métricas usadas
- **Clasificación:** Accuracy, F1 macro, ROC AUC (si es binaria), matriz de confusión y reporte por clase.
- **Regresión:** MAE, RMSE, R².

---

## 5) Comparativo y conclusiones (guía para tu README principal)
En tu README general, compara **Gradient Boosting** con tus otros dos modelos (p. ej. Random Forest y Red Neuronal):
- **Desempeño:** ¿quién dio mejor métrica? ¿en test o validación cruzada?
- **Ventajas vs Desventajas:** estabilidad, sensibilidad a hiperparámetros, tiempo de entrenamiento.
- **Aplicabilidad:** ¿qué pasaría si el dataset cambia (más ruido, más variables categóricas, más datos faltantes)?

> **Regla práctica:** si tus datos son tabulares, equilibrados y con mezclas numéricas/categóricas, Gradient Boosting suele
> ser un **excelente baseline avanzado** y, con tuning, puede superar RF/árbol y acercarse al rendimiento de XGBoost/LightGBM.

---

## 6) Cita corta sugerida (si la necesitas en el informe)
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics.

¡Listo! Abajo te dejo los archivos y el script para que lo uses tal cual.
