# Model Fitness — Churn Prediction & Customer Segmentation

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white" alt="Jupyter">
  <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

**[English](#english)** | **[Español](#español)**

---

## English

### Overview

Customer churn is one of the biggest challenges for subscription-based businesses. In this project, we analyze behavioral and demographic data from **Model Fitness**, a gym chain, to **predict which customers are likely to cancel** their memberships and **segment users into actionable groups** using machine learning.

The goal is to move from reactive retention efforts to a **proactive, data-driven strategy** that identifies at-risk customers before they leave.

### Objectives

- Predict the probability of customer churn in the following month.
- Identify typical user profiles through clustering techniques.
- Analyze the key factors that drive customer cancellations.
- Propose data-backed strategic recommendations to improve retention.

### Dataset

The dataset (`gym_churn_us.csv`) contains **4,000 records** with 14 features per customer:

| Feature | Description |
|---|---|
| `gender` | Customer gender (binary) |
| `near_location` | Whether the customer lives/works near the gym |
| `partner` | Employee of a partner company |
| `promo_friends` | Joined through a referral promotion |
| `phone` | Has a registered phone number |
| `contract_period` | Contract duration (1, 6, or 12 months) |
| `group_visits` | Participates in group classes |
| `age` | Customer age |
| `avg_additional_charges_total` | Average spending on additional services |
| `month_to_end_contract` | Months remaining on the contract |
| `lifetime` | Total months as a customer |
| `avg_class_frequency_total` | Average weekly visit frequency (lifetime) |
| `avg_class_frequency_current_month` | Average weekly visit frequency (current month) |
| `churn` | Target variable — whether the customer cancelled (1) or stayed (0) |

**Class distribution:** 26.5% churn / 73.5% non-churn.

### Methodology

The analysis follows four stages:

#### 1. Exploratory Data Analysis (EDA)

- Descriptive statistics and distribution analysis.
- Comparison of churned vs. retained customers across all features.
- Correlation matrix with multicollinearity assessment.
- Histograms segmented by churn status.

#### 2. Predictive Modeling

- Feature scaling with `StandardScaler` (fit on training data, transformed on validation data).
- **Logistic Regression** — with L2 regularization on scaled features.
- **Random Forest Classifier** — ensemble model for comparison.
- Evaluation metrics: Accuracy, Precision, Recall, and **F1 Score** (critical for imbalanced data).
- 80/20 train-validation split with `random_state=42` for reproducibility.

#### 3. Customer Segmentation (Clustering)

- Standardization of all features.
- **Hierarchical clustering** (Ward linkage) to determine optimal cluster count via dendrogram.
- **K-Means** (k=5) to segment customers into distinct behavioral groups.
- Cluster profiling through mean comparison and boxplot distributions.
- Churn rate analysis per cluster.

#### 4. Conclusions & Recommendations

- Actionable retention strategies based on cluster profiles.
- Identification of key churn drivers and intervention opportunities.

### Key Findings

**Strongest churn predictors:**
- Visit frequency (current month) — strongest negative correlation with churn.
- Customer tenure (lifetime).
- Contract duration and months remaining.
- Additional spending.

**Variables with no significant impact:** gender, phone registration.

**Clustering revealed distinct customer profiles:**
- **Premium loyal customers** (~2% churn) — long contracts, high frequency, high spending.
- **Active loyal customers** (~9% churn) — high visit frequency, long tenure.
- **High-risk customers** (~57% churn) — new, low frequency, short contracts, low spending.
- **Moderate-risk customers** (~25% churn) — intermediate behavior, potential for retention.

### Project Structure

```
Model-Fitness---Modelado-predictivo-y-Clustering/
│
├── Modelado predictivo y Clustering.ipynb   # Full analysis (Spanish)
├── Predictive Modeling and Clustering.ipynb  # Full analysis (English)
├── gym_churn_us.csv                          # Dataset
├── Análisis de Churn - Model Fitness.pdf     # Report (Spanish)
├── Model Fitness Churn Analysis.pdf          # Report (English)
├── .gitignore
└── README.md
```

### Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data manipulation | pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (LogisticRegression, RandomForestClassifier, KMeans, StandardScaler) |
| Clustering | SciPy (hierarchical clustering / dendrograms) |
| Environment | Jupyter Notebook |

### How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/DevLearnsAI/Model-Fitness---Modelado-predictivo-y-Clustering.git
   cd Model-Fitness---Modelado-predictivo-y-Clustering
   ```

2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
   ```

3. Launch the notebook:
   ```bash
   jupyter notebook "Predictive Modeling and Clustering.ipynb"
   ```

4. Run all cells sequentially to reproduce the analysis.

---

## Español

### Descripcion General

La pérdida de clientes (churn) es uno de los mayores desafíos para negocios basados en suscripciones. En este proyecto se analizan datos de comportamiento y demográficos de **Model Fitness**, una cadena de gimnasios, para **predecir qué clientes tienen mayor probabilidad de cancelar** sus membresías y **segmentar usuarios en grupos accionables** mediante machine learning.

El objetivo es pasar de estrategias de retención reactivas a una **estrategia proactiva basada en datos** que identifique clientes en riesgo antes de que se vayan.

### Objetivos

- Predecir la probabilidad de cancelación de clientes en el próximo mes.
- Identificar perfiles típicos de usuarios mediante técnicas de segmentación.
- Analizar los factores clave que impulsan la cancelación.
- Proponer recomendaciones estratégicas respaldadas por datos para mejorar la retención.

### Dataset

El dataset (`gym_churn_us.csv`) contiene **4,000 registros** con 14 características por cliente:

| Variable | Descripción |
|---|---|
| `gender` | Género del cliente (binario) |
| `near_location` | Si el cliente vive/trabaja cerca del gimnasio |
| `partner` | Empleado de una empresa asociada |
| `promo_friends` | Se unió a través de una promoción de referidos |
| `phone` | Tiene un número de teléfono registrado |
| `contract_period` | Duración del contrato (1, 6 o 12 meses) |
| `group_visits` | Participa en clases grupales |
| `age` | Edad del cliente |
| `avg_additional_charges_total` | Gasto promedio en servicios adicionales |
| `month_to_end_contract` | Meses restantes del contrato |
| `lifetime` | Total de meses como cliente |
| `avg_class_frequency_total` | Frecuencia de visitas semanal promedio (histórica) |
| `avg_class_frequency_current_month` | Frecuencia de visitas semanal promedio (mes actual) |
| `churn` | Variable objetivo — si el cliente canceló (1) o se quedó (0) |

**Distribución de clases:** 26.5% churn / 73.5% no-churn.

### Metodología

El análisis se desarrolla en cuatro etapas:

#### 1. Análisis Exploratorio de Datos (EDA)

- Estadísticas descriptivas y análisis de distribuciones.
- Comparación de clientes que cancelaron vs. los que permanecieron en todas las variables.
- Matriz de correlación con evaluación de multicolinealidad.
- Histogramas segmentados por estado de churn.

#### 2. Modelado Predictivo

- Escalado de características con `StandardScaler` (ajuste en datos de entrenamiento, transformación en validación).
- **Regresión Logística** — con regularización L2 sobre características escaladas.
- **Random Forest Classifier** — modelo de ensamble para comparación.
- Métricas de evaluación: Accuracy, Precision, Recall y **F1 Score** (fundamental con datos desbalanceados).
- Split 80/20 entrenamiento-validación con `random_state=42` para reproducibilidad.

#### 3. Segmentación de Clientes (Clustering)

- Estandarización de todas las características.
- **Clustering jerárquico** (método Ward) para determinar el número óptimo de clusters mediante dendrograma.
- **K-Means** (k=5) para segmentar clientes en grupos de comportamiento diferenciados.
- Perfilamiento de clusters mediante comparación de medias y distribuciones con boxplots.
- Análisis de tasa de churn por cluster.

#### 4. Conclusiones y Recomendaciones

- Estrategias de retención accionables basadas en los perfiles de cada cluster.
- Identificación de los principales factores de churn y oportunidades de intervención.

### Hallazgos Clave

**Predictores más fuertes de churn:**
- Frecuencia de visitas (mes actual) — correlación negativa más fuerte con churn.
- Antigüedad del cliente (lifetime).
- Duración del contrato y meses restantes.
- Gasto adicional.

**Variables sin impacto significativo:** género, registro de teléfono.

**El clustering reveló perfiles diferenciados:**
- **Clientes premium leales** (~2% churn) — contratos largos, alta frecuencia, alto gasto.
- **Clientes activos leales** (~9% churn) — alta frecuencia de visitas, alta antigüedad.
- **Clientes de alto riesgo** (~57% churn) — nuevos, baja frecuencia, contratos cortos, bajo gasto.
- **Clientes de riesgo moderado** (~25% churn) — comportamiento intermedio, potencial de fidelización.

### Estructura del Proyecto

```
Model-Fitness---Modelado-predictivo-y-Clustering/
│
├── Modelado predictivo y Clustering.ipynb   # Análisis completo (Español)
├── Predictive Modeling and Clustering.ipynb  # Análisis completo (Inglés)
├── gym_churn_us.csv                          # Dataset
├── Análisis de Churn - Model Fitness.pdf     # Reporte (Español)
├── Model Fitness Churn Analysis.pdf          # Reporte (Inglés)
├── .gitignore
└── README.md
```

### Tecnologías Utilizadas

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3 |
| Manipulación de datos | pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| Machine Learning | scikit-learn (LogisticRegression, RandomForestClassifier, KMeans, StandardScaler) |
| Clustering | SciPy (clustering jerárquico / dendrogramas) |
| Entorno | Jupyter Notebook |

### Cómo Ejecutar

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/DevLearnsAI/Model-Fitness---Modelado-predictivo-y-Clustering.git
   cd Model-Fitness---Modelado-predictivo-y-Clustering
   ```

2. Instalar dependencias:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
   ```

3. Abrir el notebook:
   ```bash
   jupyter notebook "Modelado predictivo y Clustering.ipynb"
   ```

4. Ejecutar todas las celdas en orden para reproducir el análisis.

---

<p align="center">
  <sub>Developed by <a href="https://github.com/DevLearnsAI">DevLearnsAI</a></sub>
</p>
