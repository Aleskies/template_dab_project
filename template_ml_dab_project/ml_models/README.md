# ML Models Module

Este módulo contiene todos los componentes necesarios para entrenar, evaluar y desplegar modelos de Machine Learning en Databricks.

## 📋 Estructura

```
ml_models/
├── __init__.py              # Inicialización del módulo
├── train.py                 # Script principal de entrenamiento
├── predict.py               # Script de predicción
├── feature_engineering.py   # Ingeniería de características
├── model_config.py          # Configuración de modelos
├── mlflow_utils.py          # Utilidades MLflow
└── README.md               # Esta documentación
```

## 🚀 Componentes Principales

### 1. model_config.py
Configuración centralizada para modelos ML.

**Características:**
* Configuración de datos (catálogo, esquema, tablas)
* Configuración de features y target
* Hiperparámetros del modelo
* Configuración de MLflow
* Configuraciones predefinidas (clasificación, regresión, XGBoost)

**Uso:**
```python
from ml_models.model_config import ModelConfig, CLASSIFICATION_CONFIG

# Usar configuración predefinida
config = CLASSIFICATION_CONFIG

# O crear configuración personalizada
config = ModelConfig(
    model_name="my_custom_model",
    catalog="workspace",
    schema="my_schema",
    train_table="my_train_data",
    feature_columns=["col1", "col2", "col3"],
    target_column="target"
)
```

### 2. feature_engineering.py
Clase `FeatureEngineer` con funciones para preparación de datos.

**Funcionalidades:**
* Manejo de valores faltantes
* Escalado de features (StandardScaler, MinMaxScaler)
* Codificación de variables categóricas (Label, OneHot)
* Creación de features de fecha
* Features de interacción y polinomiales
* Features de agregación
* Detección y eliminación de outliers
* Análisis de importancia de features

**Uso:**
```python
from ml_models.feature_engineering import FeatureEngineer

fe = FeatureEngineer()

# Manejar valores faltantes
df = fe.handle_missing_values(df, strategy="mean")

# Escalar features
df = fe.scale_features(df, columns=["age", "income"], method="standard")

# Codificar categóricas
df = fe.encode_categorical(df, columns=["category"], method="onehot")

# Crear features de fecha
df = fe.create_date_features(df, date_column="timestamp")
```

### 3. mlflow_utils.py
Clase `MLflowTracker` para tracking de experimentos.

**Funcionalidades:**
* Gestión de experimentos MLflow
* Logging de parámetros, métricas y artefactos
* Registro de modelos en Model Registry
* Gestión de versiones y aliases
* Búsqueda y comparación de runs
* Carga de modelos

**Uso:**
```python
from ml_models.mlflow_utils import MLflowTracker

tracker = MLflowTracker(experiment_name="/Shared/my_experiment")

with tracker.start_run(run_name="training_v1"):
    # Log parámetros
    tracker.log_params({"learning_rate": 0.01, "max_depth": 10})
    
    # Log métricas
    tracker.log_metrics({"accuracy": 0.95, "f1_score": 0.93})
    
    # Log modelo
    tracker.log_model(
        model,
        artifact_path="model",
        model_type="sklearn",
        registered_model_name="my_model"
    )
    
    # Set alias
    tracker.set_model_alias("my_model", "Production", version=1)
```

### 4. train.py
Script principal para entrenamiento de modelos.

**Características:**
* Carga de datos desde Unity Catalog
* Preparación automática de features
* Entrenamiento con MLflow tracking
* Evaluación y métricas
* Registro automático en Model Registry
* Análisis de importancia de features

**Uso desde línea de comandos:**
```bash
python ml_models/train.py \
    --catalog workspace \
    --schema my_schema \
    --train-table train_data \
    --model-name my_ml_model \
    --task-type classification \
    --register
```

**Uso programático:**
```python
from ml_models.train import ModelTrainer
from ml_models.model_config import ModelConfig

config = ModelConfig(
    catalog="workspace",
    schema="my_schema",
    train_table="train_data",
    model_name="my_model"
)

trainer = ModelTrainer(config)
trainer.run_training_pipeline(
    task_type="classification",
    register_model=True
)
```

### 5. predict.py
Script para hacer predicciones con modelos entrenados.

**Características:**
* Carga de modelos desde Model Registry
* Predicciones individuales o por lotes
* Predicciones probabilísticas
* Batch processing de tablas completas
* Explicaciones de predicciones

**Uso desde línea de comandos:**
```bash
python ml_models/predict.py \
    --model-name my_ml_model \
    --model-version Production \
    --input-table workspace.my_schema.test_data \
    --output-table workspace.my_schema.predictions \
    --batch-size 1000
```

**Uso programático:**
```python
from ml_models.predict import ModelPredictor

# Cargar modelo
predictor = ModelPredictor(
    model_name="my_ml_model",
    model_version="Production"
)

# Predicción individual
prediction = predictor.predict({
    "feature1": 1.0,
    "feature2": 2.0,
    "feature3": 3.0
})

# Predicciones probabilísticas
probabilities = predictor.predict_proba({
    "feature1": 1.0,
    "feature2": 2.0,
    "feature3": 3.0
})

# Batch predictions
predictor.batch_predict(
    input_table="workspace.my_schema.test_data",
    output_table="workspace.my_schema.predictions",
    batch_size=1000
)
```

## 📊 Pipeline Completo de ML

### 1. Preparación de Datos

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Crear tabla de entrenamiento
df = spark.sql("""
    SELECT 
        feature1,
        feature2,
        feature3,
        target
    FROM source_table
    WHERE date >= '2024-01-01'
""")

df.write.mode("overwrite").saveAsTable("workspace.my_schema.train_data")
```

### 2. Entrenamiento

```python
from ml_models.train import ModelTrainer
from ml_models.model_config import ModelConfig

# Configuración
config = ModelConfig(
    catalog="workspace",
    schema="my_schema",
    train_table="train_data",
    model_name="my_ml_model",
    feature_columns=["feature1", "feature2", "feature3"],
    target_column="target",
    model_type="random_forest",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": 42
    }
)

# Entrenar
trainer = ModelTrainer(config)
trainer.run_training_pipeline(
    task_type="classification",
    register_model=True
)
```

### 3. Predicción

```python
from ml_models.predict import ModelPredictor

# Cargar modelo
predictor = ModelPredictor(
    model_name="my_ml_model",
    model_version="Production"
)

# Hacer predicciones
predictor.batch_predict(
    input_table="workspace.my_schema.new_data",
    output_table="workspace.my_schema.predictions"
)
```

## 🔧 Configuración Avanzada

### Configurar Experimento MLflow

```python
config = ModelConfig(
    experiment_name="/Shared/ml_experiments/my_project",
    model_registry_name="my_project_model",
    model_alias="Production"
)
```

### Configurar Features Personalizadas

```python
from ml_models.feature_engineering import FeatureEngineer

fe = FeatureEngineer()

# Crear features complejas
df = fe.create_date_features(df, "timestamp")
df = fe.create_interaction_features(df, [("feature1", "feature2")])
df = fe.create_polynomial_features(df, ["feature1", "feature2"], degree=2)
df = fe.create_aggregation_features(
    df, 
    group_column="category",
    agg_columns=["value"],
    agg_functions=["mean", "std", "max"]
)
```

### Comparar Diferentes Modelos

```python
from ml_models.mlflow_utils import MLflowTracker

tracker = MLflowTracker("/Shared/ml_experiments")

# Buscar runs
runs_df = tracker.search_runs(
    filter_string="metrics.accuracy > 0.9",
    max_results=10
)

# Comparar runs
comparison = tracker.compare_runs(
    run_ids=["run_id_1", "run_id_2"],
    metric_names=["accuracy", "f1_score", "precision"]
)

# Obtener mejor run
best_run = tracker.get_best_run(
    metric_name="accuracy",
    ascending=False
)
```

## 🎯 Tipos de Modelos Soportados

### 1. Random Forest (por defecto)
```python
config = ModelConfig(
    model_type="random_forest",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2
    }
)
```

### 2. XGBoost
```python
config = ModelConfig(
    model_type="xgboost",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    }
)
```

### 3. LightGBM
```python
config = ModelConfig(
    model_type="lightgbm",
    hyperparameters={
        "n_estimators": 100,
        "max_depth": -1,
        "learning_rate": 0.1
    }
)
```

## 📈 Integración con Databricks Jobs

Crear un job en `resources/ml_training.job.yml`:

```yaml
resources:
  jobs:
    ml_training_job:
      name: "ML Training Job"
      tasks:
        - task_key: train_model
          python_task:
            python_file: ml_models/train.py
            parameters:
              - "--catalog"
              - "workspace"
              - "--schema"
              - "${var.schema}"
              - "--train-table"
              - "train_data"
              - "--model-name"
              - "my_ml_model"
              - "--task-type"
              - "classification"
              - "--register"
          new_cluster:
            spark_version: "14.3.x-scala2.12"
            node_type_id: "i3.xlarge"
            num_workers: 2
      schedule:
        quartz_cron_expression: "0 0 2 * * ?"
        timezone_id: "America/Los_Angeles"
```

## 🔗 Integración con Flask App

Los modelos entrenados se integran automáticamente con la aplicación Flask:

1. El modelo se registra en MLflow con alias "Production"
2. La Flask app carga el modelo desde MLflow
3. Los endpoints exponen predicciones vía API REST

Ver `flask_app/README.md` para más detalles.

## 📝 Best Practices

* **Usa Unity Catalog** para todas las tablas de datos
* **Versionea tus modelos** con aliases claros (Production, Staging, etc.)
* **Trackea todos los experimentos** con MLflow
* **Valida features** antes del entrenamiento
* **Documenta hiperparámetros** en ModelConfig
* **Evalúa en datos de test** antes de promocionar a Production
* **Monitorea performance** del modelo en producción

## 🐛 Troubleshooting

### Error: "Table not found"
Verifica que la tabla existe en Unity Catalog:
```python
spark.sql("SHOW TABLES IN workspace.my_schema").show()
```

### Error: "Model not found in registry"
Lista modelos registrados:
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
models = client.search_registered_models()
for model in models:
    print(model.name)
```

### Error: "Feature column not found"
Verifica columnas en tu tabla:
```python
df = spark.table("workspace.my_schema.train_data")
df.printSchema()
```
