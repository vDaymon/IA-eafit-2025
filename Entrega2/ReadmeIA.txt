Actividad 2 – Machine Learning Supervisado

Proyecto: Titanic — Clasificación (Supervivencia)

1) Descripción del dataset
	•	Fuente: Kaggle — “Titanic: Machine Learning from Disaster”.
	•	Objetivo: Clasificar si un pasajero sobrevivió (1) o no sobrevivió (0).
	•	Archivo utilizado: train.csv (etiquetado).
	•	Tamaño aproximado: ~891 registros.
	•	Variables originales (principales): PassengerId, Survived (target), Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.
	•	Variables efectivas tras preprocesar (features): Pclass, Age, SibSp, Parch, Fare, Sex_male, Embarked_Q, Embarked_S.
	•	Ruta de datos utilizada en el script:
C:\Users\USUARIO\OneDrive\Documentos\TrabajoIA\titanic\train.csv (esto se debe ajustar al descargar el programa y debes poner la ruta en donde esté el archivo “train.cvs”)


2) Preprocesamiento realizado

Se aplicó el mismo pipeline a los tres modelos para una comparación justa:

a) Limpieza / imputación de faltantes
	•	Columnas eliminadas por baja utilidad y/o alta ausencia: PassengerId, Name, Ticket, Cabin.
	•	Imputaciones:
	•	Age → mediana
	•	Embarked → moda

b) Codificación de categóricas
	•	One-Hot Encoding con pd.get_dummies(..., drop_first=True) en Sex y Embarked.
	•	Resultan, por ejemplo, Sex_male, Embarked_Q, Embarked_S.

c) Escalado / normalización
	•	StandardScaler aplicado después del split (para evitar fuga de información):
	•	X_train_s = scaler.fit_transform(X_train)
	•	X_test_s  = scaler.transform(X_test)
	•	Justificación: MLP requiere escalado; en Random Forest y Gradient Boosting no es necesario, pero no afecta negativamente.

d) División en train/test
	•	train_test_split(test_size=0.20, random_state=42, stratify=y)
	•	Estratificación para mantener la proporción de clases.

3) Modelos y parámetros (entrenamiento)

Los tres modelos se entrenaron sobre el mismo conjunto preprocesado:

3.1 Modelo clásico (elegido): Random Forest

RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

3.2 Red Neuronal (MLPClassifier)
MLPClassifier(
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

3.3 Algoritmo adicional (investigado): Gradient Boosting

GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

4) Evaluación de resultados (métricas y visualizaciones)

Se usó la función evaluate_model(...) para entrenar y evaluar cada modelo en el mismo conjunto de prueba.
Se reportan: Accuracy, Precision, Recall, F1, ROC-AUC, reporte de clasificación y matriz de confusión.
Además se imprimen importancias para RF y GB, y una tabla comparativa final.

4.1 Métricas por modelo (valores reales del output)

Random Forest
	•	Accuracy: 0.8156
	•	Reporte de clasificación:
	•	Clase 0 → precision 0.83, recall 0.87, f1 0.85 (support 110)
	•	Clase 1 → precision 0.78, recall 0.72, f1 0.75 (support 69)
	•	Macro avg → precision 0.81, recall 0.80, f1 0.80
	•	Weighted avg → precision 0.81, recall 0.82, f1 0.81

•	Matriz de confusión: 

[96 14]
[19 50]

•	ROC-AUC: 0.8314

Red Neuronal (MLP)
	•	Accuracy: 0.8101
	•	Reporte de clasificación:
	•	Clase 0 → precision 0.81, recall 0.91, f1 0.85 (support 110)
	•	Clase 1 → precision 0.82, recall 0.65, f1 0.73 (support 69)
	•	Macro avg → precision 0.81, recall 0.78, f1 0.79
	•	Weighted avg → precision 0.81, recall 0.81, f1 0.81
	•	Matriz de confusión:
[100  10]
[ 24  45]

	•	ROC-AUC: 0.8576

Gradient Boosting
	•	Accuracy: 0.7989
	•	Reporte de clasificación:
	•	Clase 0 → precision 0.80, recall 0.89, f1 0.84 (support 110)
	•	Clase 1 → precision 0.79, recall 0.65, f1 0.71 (support 69)
	•	Macro avg → precision 0.80, recall 0.77, f1 0.78
	•	Weighted avg → precision 0.80, recall 0.80, f1 0.79
	•	Matriz de confusión:

[98 12]
[24 45]

•	ROC-AUC: 0.8247

4.2 Importancia de características

Random Forest (feature_importances_)
	1.	Fare 0.2785
	2.	Sex_male 0.2632
	3.	Age 0.2524
	4.	Pclass 0.0795
	5.	SibSp 0.0535
	6.	Parch 0.0410
	7.	Embarked_S 0.0232
	8.	Embarked_Q 0.0087

Gradient Boosting (feature_importances_)
	1.	Sex_male 0.4628
	2.	Fare 0.1856
	3.	Pclass 0.1507
	4.	Age 0.1493
	5.	SibSp 0.0274
	6.	Embarked_S 0.0219
	7.	Parch 0.0018
	8.	Embarked_Q 0.0005

5) Análisis comparativo

Desempeño general:
	•	Accuracy: RF (0.8156) ≈ MLP (0.8101) > GB (0.7989).
	•	ROC-AUC: MLP lidera (0.8576), luego RF (0.8314), después GB (0.8247).
	•	Clase 1 (sobrevivió) – Recall/F1:
	•	MLP y GB tienen recall 0.65 (igual), RF logra 0.72 en clase 1 (mejor recall de positivos).
	•	En F1 de clase positiva, RF ≈ 0.75 > MLP ≈ 0.73 > GB ≈ 0.71.
	•	Interpretabilidad:
	•	RF y GB entregan importancias: sexo, tarifa, edad y clase social encabezan (coherente con contexto histórico del Titanic).
	•	MLP no ofrece interpretabilidad directa, pero logra el mejor ROC-AUC, útil si interesa el ranking/probabilidad.

Ventajas y desventajas (resumen):
	•	Random Forest (clásico elegido): robusto, buen balance y mejor F1/recall en clase 1; interpretabilidad por importancias.
	•	MLP: mejor ROC-AUC (mejor ranking de positivos vs negativos), pero menos interpretable; requiere escalado y buen ajuste.
	•	Gradient Boosting: competitivo, a veces gana con tuning fino; aquí quedó levemente por debajo en accuracy/F1.

Escenarios de aplicación:
	•	Si prima interpretabilidad + buen desempeño global, RF es la elección natural.
	•	Si se requiere mejor discriminación probabilística (p.ej., para ordenar por riesgo y usar umbrales ajustables), MLP destaca en ROC-AUC.
	•	Con mayor tuning (p.ej., learning_rate, n_estimators, max_depth), GB puede mejorar y competir cabeza a cabeza.

6) Conclusiones
	•	El modelo clásico (Random Forest) cumple el rol solicitado y ofrece el mejor equilibrio entre performance (accuracy 0.8156; F1 y recall de la clase positiva superiores) e interpretabilidad (importancias).
	•	La Red Neuronal (MLP) logra la mejor ROC-AUC (0.8576), lo que puede ser preferible si el objetivo es rankear por probabilidad y ajustar umbrales más tarde.
	•	Gradient Boosting mostró un desempeño ligeramente menor en este setup, pero con tuning apropiado podría igualar o superar; sigue siendo una alternativa sólida en tabulares.
	•	Para este dataset y configuración, recomendamos Random Forest como modelo operativo por su equilibrio y transparencia; MLP como segunda opción cuando se requiera mejor ranking probabilístico.

7) Reproducibilidad

Ejecución:
	1.	Ajustar variable df que tiene la ruta del archivo train.csv.
	2.	Instalar dependencias mínimas:
	•	pandas, numpy, scikit-learn (Comando de instalación desde la terminal de la librería usada: “pip install scikit-learn pandas”)
 
3.	Ejecutar el script principal y verificar la consola para métricas e importancias.

Nota: Los resultados anteriores puestos en este archivo son los arrojados por el programa adjunto al terminar de compilar  (con random_state=42 y test_size=0.20).
