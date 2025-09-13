# Titanic – Random Forest (Aprendizaje Supervisado)

Proyecto de IA (supervisado) que entrena un **Random Forest** para predecir la supervivencia de pasajeros del Titanic.

## 1) Dataset

Usamos el clásico dataset **Titanic** de Kaggle (CSV):

- Página: Titanic – Machine Learning from Disaster
- Archivos relevantes: `train.csv` (incluye la etiqueta `Survived`), `test.csv` (sin etiquetas, opcional)

**Cómo descargar:**
1. Crear cuenta en Kaggle (gratis).
2. Ir a la página del concurso y pulsar **Download All**, link "https://www.kaggle.com/c/titanic/data?utm_source"".
3. Descomprimir y ubicar `train.csv` en tu carpeta del proyecto.

> En este repo se asume la ruta:
> ```
> C:\Users\USUARIO\OneDrive\Documentos\TrabajoIA\titanic\train.csv
> ```

## 2) Requisitos y entorno

- Python 3.10+ (probado en 3.13)
- Librerías:
  - `pandas` (manipulación de datos)
  - `scikit-learn` (modelado y métricas)
  - `numpy` (soporte numérico)

**Instalación (global o en venv):**
pip install scikit-learn pandas
