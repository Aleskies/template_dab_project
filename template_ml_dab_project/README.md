# Template ML DAB Project

Plantilla completa de proyecto de Machine Learning con aplicación Flask para Databricks, utilizando Declarative Automation Bundles (DABs).

## 🎯 Descripción

Este proyecto proporciona una arquitectura completa para desarrollar, entrenar, desplegar y servir modelos de Machine Learning en Databricks, incluyendo:

* **Pipeline de ML** completo (feature engineering, training, prediction)
* **Aplicación Flask** para servir modelos vía API REST
* **Integración con MLflow** para experiment tracking y model registry
* **Configuración DABs** para deployment automatizado
* **Jobs programables** para entrenamiento y predicción batch

## 📁 Estructura del Proyecto

```
template_ml_dab_project/
├── databricks.yml                 # Configuración principal del bundle
├── pyproject.toml                 # Dependencias Python
├── README.md                      # Esta documentación
│
├── src/                           # Código fuente del paquete
│   └── template_ml_dab_project/
│       ├── __init__.py
│       ├── main.py
│       └── taxis.py
│
├── ml_models/                     # 🤖 Módulo de Machine Learning
│   ├── __init__.py
│   ├── train.py                   # Script de entrenamiento
│   ├── predict.py                 # Script de predicción
│   ├── feature_engineering.py     # Feature engineering
│   ├── model_config.py            # Configuración de modelos
│   ├── mlflow_utils.py            # Utilidades MLflow
│   └── README.md                  # Documentación del módulo ML
│
├── flask_app/                     # 🌐 Aplicación Flask (Databricks App)
│   ├── app.py                     # Aplicación principal
│   ├── requirements.txt           # Dependencias Flask
│   ├── app.yml                    # Configuración Databricks App
│   ├── README.md                  # Documentación de la app
│   ├── templates/
│   │   └── index.html             # UI con documentación API
│   └── static/
│       └── style.css              # Estilos CSS
│
├── resources/                     # Definiciones de recursos DABs
│   ├── sample_job.job.yml         # Job de ejemplo
│   └── ml_training.job.yml        # Jobs de ML (training & prediction)
│
├── tests/                         # Tests unitarios
│   ├── conftest.py
│   └── sample_taxis_test.py
│
└── fixtures/                      # Datos de prueba
```

## 🚀 Quick Start

### 1. Preparar Datos

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()

# Crear tabla de entrenamiento
spark.sql("""
    CREATE TABLE workspace.{schema}.train_data AS
    SELECT 
        feature1,
        feature2,
        feature3,
        target
    FROM source_table
""")
```

### 2. Entrenar Modelo

```python
from ml_models.train import ModelTrainer
from ml_models.model_config import ModelConfig

config = ModelConfig(
    catalog="workspace",
    schema="my_schema",
    train_table="train_data",
    model_name="my_ml_model",
    feature_columns=["feature1", "feature2", "feature3"],
    target_column="target"
)

trainer = ModelTrainer(config)
trainer.run_training_pipeline(
    task_type="classification",
    register_model=True
)
```

### 3. Desplegar Aplicación Flask

```bash
# Validar configuración
databricks bundle validate --target dev

# Desplegar todo (jobs + app)
databricks bundle deploy --target dev

# Ejecutar app Flask
databricks bundle run ml_flask_app --target dev
```

### 4. Hacer Predicciones

**Via Python:**
```python
from ml_models.predict import ModelPredictor

predictor = ModelPredictor("my_ml_model", "Production")
prediction = predictor.predict({"feature1": 1.0, "feature2": 2.0})
```

**Via API REST:**
```bash
curl -X POST https://your-app-url/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0, "feature3": 3.0}'
```

## 🤖 Módulo ML (`ml_models/`)

### Componentes

* **`model_config.py`**: Configuración centralizada (datos, features, hiperparámetros)
* **`feature_engineering.py`**: Preparación y transformación de datos
* **`mlflow_utils.py`**: Tracking de experimentos y model registry
* **`train.py`**: Pipeline completo de entrenamiento
* **`predict.py`**: Predicciones individuales y batch

### Ejemplo de Entrenamiento

```python
# Desde línea de comandos
python ml_models/train.py \
    --catalog workspace \
    --schema my_schema \
    --train-table train_data \
    --model-name my_model \
    --task-type classification \
    --register

# Programático
from ml_models.train import ModelTrainer
from ml_models.model_config import CLASSIFICATION_CONFIG

trainer = ModelTrainer(CLASSIFICATION_CONFIG)
trainer.run_training_pipeline(task_type="classification", register_model=True)
```

Ver documentación completa en [`ml_models/README.md`](ml_models/README.md)

## 🌐 Aplicación Flask (`flask_app/`)

### Características

* API REST para predicciones
* UI interactiva con documentación
* Carga automática de modelos desde MLflow
* Health checks y monitoring
* Lista para Databricks Apps

### Endpoints

* `GET /` - Página principal con documentación
* `GET /health` - Health check
* `GET /model-info` - Información del modelo
* `POST /predict` - Hacer predicción

### Configuración

Edita `flask_app/app.yml` para configurar:

```yaml
variables:
  model_name:
    default: "my_ml_model"
  
  model_version:
    default: "Production"
  
  warehouse_id:
    default: "your_warehouse_id"
```

Ver documentación completa en [`flask_app/README.md`](flask_app/README.md)

## ⚙️ Configuración DABs

### Archivo Principal: `databricks.yml`

```yaml
bundle:
  name: template_ml_dab_project

include:
  - resources/*.yml
  - flask_app/app.yml

variables:
  catalog:
    description: The catalog to use
  schema:
    description: The schema to use

targets:
  dev:
    mode: development
    default: true
    variables:
      catalog: workspace
      schema: ${workspace.current_user.short_name}
  
  prod:
    mode: production
    variables:
      catalog: workspace
      schema: prod
```

### Resources: `resources/ml_training.job.yml`

Define jobs para:
* **Training**: Entrena modelos automáticamente
* **Batch Prediction**: Predicciones en lotes sobre tablas

## 🔄 Flujo de Trabajo Completo

### 1. Desarrollo

```bash
# 1. Preparar datos
python -c "from setup_data import create_sample_data; create_sample_data()"

# 2. Entrenar modelo localmente
python ml_models/train.py --catalog workspace --schema dev --register

# 3. Probar predicciones
python ml_models/predict.py --model-name my_model --model-version Production
```

### 2. Deployment

```bash
# Validar bundle
databricks bundle validate --target dev

# Desplegar a dev
databricks bundle deploy --target dev

# Ejecutar job de training
databricks bundle run ml_training_job --target dev

# Desplegar app Flask
databricks bundle run ml_flask_app --target dev
```

### 3. Producción

```bash
# Desplegar a prod
databricks bundle deploy --target prod

# Jobs se ejecutan automáticamente según schedule
# App Flask queda disponible para requests
```

## 📊 Integración MLflow

### Tracking de Experimentos

```python
from ml_models.mlflow_utils import MLflowTracker

tracker = MLflowTracker("/Shared/ml_experiments")

with tracker.start_run():
    tracker.log_params({"learning_rate": 0.01})
    tracker.log_metrics({"accuracy": 0.95})
    tracker.log_model(model, registered_model_name="my_model")
```

### Model Registry

```python
# Set alias para versión
tracker.set_model_alias("my_model", "Production", version=3)

# Cargar modelo por alias
model = tracker.load_model("my_model", alias="Production")

# Comparar versiones
comparison = tracker.compare_runs(
    run_ids=["run1", "run2"],
    metric_names=["accuracy", "f1_score"]
)
```

## 🧪 Testing

```bash
# Ejecutar tests
pytest tests/

# Con coverage
pytest tests/ --cov=ml_models --cov=flask_app
```

## 📝 Variables de Entorno

Para desarrollo local, crea un archivo `.env`:

```bash
# MLflow
MODEL_NAME=my_ml_model
MODEL_VERSION=Production
MLFLOW_TRACKING_URI=databricks

# Databricks
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your-token

# Data
CATALOG=workspace
SCHEMA=my_schema
```

## 🎓 Ejemplos de Uso

### Feature Engineering Avanzado

```python
from ml_models.feature_engineering import FeatureEngineer

fe = FeatureEngineer()

# Pipeline completo
df = fe.handle_missing_values(df)
df = fe.create_date_features(df, "timestamp")
df = fe.create_interaction_features(df, [("price", "quantity")])
df = fe.scale_features(df, ["age", "income"])
df = fe.encode_categorical(df, ["category", "region"])
```

### Configuraciones Predefinidas

```python
from ml_models.model_config import (
    CLASSIFICATION_CONFIG,
    REGRESSION_CONFIG,
    XGBOOST_CONFIG
)

# Usar configuración predefinida
trainer = ModelTrainer(XGBOOST_CONFIG)
```

### Batch Predictions

```python
predictor = ModelPredictor("my_model", "Production")

predictor.batch_predict(
    input_table="workspace.schema.new_data",
    output_table="workspace.schema.predictions",
    batch_size=1000
)
```

## 🔧 Personalización

### Añadir Nuevos Modelos

Edita `ml_models/model_config.py`:

```python
from dataclasses import dataclass

@dataclass
class CustomModelConfig(ModelConfig):
    model_type: str = "custom"
    hyperparameters: Dict[str, Any] = field(default_factory=lambda: {
        "param1": value1,
        "param2": value2
    })
```

### Añadir Nuevos Endpoints Flask

Edita `flask_app/app.py`:

```python
@app.route("/custom-endpoint", methods=["POST"])
def custom_endpoint():
    # Tu lógica aquí
    return jsonify({"result": "success"})
```

## 📚 Documentación Adicional

* [ML Models Module](ml_models/README.md) - Documentación detallada del módulo ML
* [Flask Application](flask_app/README.md) - Documentación de la aplicación Flask
* [Databricks DABs](https://docs.databricks.com/dev-tools/bundles/index.html) - Documentación oficial de DABs
* [MLflow](https://mlflow.org/docs/latest/index.html) - Documentación de MLflow

## 🐛 Troubleshooting

### Problema: "Table not found"
**Solución**: Verifica que las tablas existen en Unity Catalog
```python
spark.sql("SHOW TABLES IN workspace.my_schema").show()
```

### Problema: "Model not found in registry"
**Solución**: Lista modelos registrados
```python
from mlflow.tracking import MlflowClient
client = MlflowClient()
for model in client.search_registered_models():
    print(model.name)
```

### Problema: "Bundle validation failed"
**Solución**: Verifica sintaxis YAML y variables
```bash
databricks bundle validate --target dev
```

## 🤝 Contributing

1. Crea una rama para tu feature: `git checkout -b feature/nueva-funcionalidad`
2. Commit tus cambios: `git commit -am 'Add nueva funcionalidad'`
3. Push a la rama: `git push origin feature/nueva-funcionalidad`
4. Crea un Pull Request

## 📄 License

Este proyecto es una plantilla para uso interno.

## 👥 Autores

* **Template** - Databricks Default Python Template
* **ML Module & Flask App** - Extensión personalizada

## 🙏 Agradecimientos

* Databricks por la plataforma y herramientas
* MLflow por el experiment tracking
* Flask por el framework web
