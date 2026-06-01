# 🚀 Quick Start Guide

Guía rápida para comenzar a usar el template de ML con Flask App en Databricks.

## ⚡ En 5 Minutos

### 1. Crear Datos de Ejemplo (Opcional)

Si no tienes datos, genera datos sintéticos para probar:

```python
# En un notebook Databricks
from sklearn.datasets import make_classification
import pandas as pd

# Crear datos
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
df = pd.DataFrame(X, columns=[f"feature{i+1}" for i in range(10)])
df['target'] = y

# Guardar en Unity Catalog
spark.createDataFrame(df).write.mode("overwrite").saveAsTable("workspace.{tu_schema}.train_data")
```

### 2. Entrenar tu Primer Modelo

```python
from ml_models.train import ModelTrainer
from ml_models.model_config import ModelConfig

config = ModelConfig(
    catalog="workspace",
    schema="{tu_schema}",  # Reemplaza con tu schema
    train_table="train_data",
    model_name="my_first_model",
    feature_columns=["feature1", "feature2", "feature3", "feature4", "feature5",
                    "feature6", "feature7", "feature8", "feature9", "feature10"],
    target_column="target"
)

trainer = ModelTrainer(config)
trainer.run_training_pipeline(task_type="classification", register_model=True)
```

### 3. Hacer Predicciones

```python
from ml_models.predict import ModelPredictor

predictor = ModelPredictor("my_first_model", "Production")

# Predicción individual
result = predictor.predict({
    "feature1": 0.5, "feature2": -1.2, "feature3": 0.8,
    "feature4": 1.5, "feature5": -0.3, "feature6": 0.1,
    "feature7": 2.0, "feature8": -0.5, "feature9": 1.1,
    "feature10": 0.7
})

print(f"Prediction: {result}")
```

## 📦 Deployment con DABs

### 1. Validar Configuración

```bash
databricks bundle validate --target dev
```

### 2. Desplegar

```bash
databricks bundle deploy --target dev
```

### 3. Ejecutar Job de Training

```bash
databricks bundle run ml_training_job --target dev
```

### 4. Desplegar Flask App

```bash
databricks bundle run ml_flask_app --target dev
```

## 🌐 Usando la Flask App

Una vez desplegada, la app expone estos endpoints:

**Health Check:**
```bash
curl https://your-app-url/health
```

**Hacer Predicción:**
```bash
curl -X POST https://your-app-url/predict \
  -H "Content-Type: application/json" \
  -d '{
    "feature1": 0.5,
    "feature2": -1.2,
    "feature3": 0.8,
    "feature4": 1.5,
    "feature5": -0.3,
    "feature6": 0.1,
    "feature7": 2.0,
    "feature8": -0.5,
    "feature9": 1.1,
    "feature10": 0.7
  }'
```

## 🎯 Casos de Uso Comunes

### Caso 1: Clasificación Binaria

```python
from ml_models.model_config import CLASSIFICATION_CONFIG

config = CLASSIFICATION_CONFIG
config.catalog = "workspace"
config.schema = "my_schema"
config.train_table = "customer_churn"
config.feature_columns = ["age", "tenure", "monthly_charges", "total_charges"]
config.target_column = "churned"
config.model_name = "churn_predictor"

trainer = ModelTrainer(config)
trainer.run_training_pipeline(task_type="classification", register_model=True)
```

### Caso 2: Regresión

```python
from ml_models.model_config import REGRESSION_CONFIG

config = REGRESSION_CONFIG
config.catalog = "workspace"
config.schema = "my_schema"
config.train_table = "house_prices"
config.feature_columns = ["sqft", "bedrooms", "bathrooms", "age"]
config.target_column = "price"
config.model_name = "price_predictor"

trainer = ModelTrainer(config)
trainer.run_training_pipeline(task_type="regression", register_model=True)
```

### Caso 3: Batch Predictions

```python
from ml_models.predict import ModelPredictor

predictor = ModelPredictor("price_predictor", "Production")

predictor.batch_predict(
    input_table="workspace.my_schema.new_houses",
    output_table="workspace.my_schema.predicted_prices",
    batch_size=1000
)
```

## 🔧 Configuración de Variables

Antes de desplegar a producción, configura las variables en `databricks.yml`:

```yaml
targets:
  prod:
    mode: production
    variables:
      catalog: prod_catalog
      schema: ml_models
```

Y en `flask_app/app.yml`:

```yaml
variables:
  model_name:
    default: "my_production_model"
  
  model_version:
    default: "Production"
  
  warehouse_id:
    default: "abc123..."  # Tu Warehouse ID
```

## 📊 Monitoreo con MLflow

Ver experimentos en MLflow UI:

1. Abre Databricks workspace
2. Navega a "Machine Learning" > "Experiments"
3. Busca tu experimento (default: `/Shared/ml_experiments`)
4. Explora runs, métricas y modelos registrados

## 🐛 Troubleshooting Rápido

**Error: "Table not found"**
```python
# Verificar tablas disponibles
spark.sql("SHOW TABLES IN workspace.my_schema").show()
```

**Error: "Model not found"**
```python
# Listar modelos registrados
import mlflow
client = mlflow.tracking.MlflowClient()
for m in client.search_registered_models():
    print(f"Model: {m.name}")
```

**Error: Bundle validation failed**
```bash
# Ver detalles del error
databricks bundle validate --target dev --verbose
```

## 📚 Próximos Pasos

1. **Personaliza tu modelo**: Edita `ml_models/model_config.py` con tus features
2. **Añade feature engineering**: Usa `ml_models/feature_engineering.py`
3. **Customiza la app**: Modifica `flask_app/app.py` y templates
4. **Programa jobs**: Descomenta el schedule en `resources/ml_training.job.yml`
5. **Lee la documentación completa**: Consulta `README.md` y los READMEs de cada módulo

## 💡 Tips

* Usa `${var.schema}` en lugar de hardcodear el schema
* Siempre registra modelos con aliases (`Production`, `Staging`)
* Valida el bundle antes de cada deploy
* Revisa los logs en Databricks UI si algo falla
* Comienza en `dev`, prueba bien, luego despliega a `prod`

## 🆘 Ayuda

* **Documentación completa**: `README.md`
* **Módulo ML**: `ml_models/README.md`
* **Flask App**: `flask_app/README.md`
* **Databricks Docs**: https://docs.databricks.com/dev-tools/bundles/
