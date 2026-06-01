# Flask Application for Databricks Apps

Esta carpeta contiene una aplicación Flask lista para ser desplegada como Databricks App, diseñada para servir modelos de ML a través de una API REST.

## 📋 Estructura

```
flask_app/
├── app.py              # Aplicación principal Flask
├── requirements.txt    # Dependencias Python
├── app.yml            # Configuración Databricks App
├── README.md          # Esta documentación
├── templates/         # Templates HTML
│   └── index.html     # Página principal con documentación de API
└── static/            # Archivos estáticos
    └── style.css      # Estilos CSS
```

## 🚀 Características

* **API REST** para predicciones de modelos ML
* **Integración con MLflow** para cargar modelos registrados
* **Interfaz web** con documentación interactiva
* **Health checks** para monitoreo
* **Configuración flexible** mediante variables de entorno

## 📡 Endpoints Disponibles

### GET /
Página principal con documentación de la API y prueba interactiva

### GET /health
Health check del servicio
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00",
  "model_loaded": true
}
```

### GET /model-info
Información del modelo cargado
```json
{
  "model_name": "my_ml_model",
  "model_version": "Production",
  "model_metadata": {...}
}
```

### POST /predict
Realizar predicciones
```bash
curl -X POST https://your-app-url/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 1.0, "feature2": 2.0}'
```

Respuesta:
```json
{
  "prediction": [result],
  "timestamp": "2024-01-01T12:00:00",
  "model": "my_ml_model",
  "version": "Production"
}
```

## ⚙️ Configuración

### Variables de Entorno

* `MODEL_NAME`: Nombre del modelo en MLflow (default: "my_ml_model")
* `MODEL_VERSION`: Versión o alias del modelo (default: "Production")
* `DATABRICKS_TOKEN`: Token de autenticación (opcional)

### Archivo app.yml

Configura las variables en `app.yml`:

```yaml
variables:
  model_name:
    default: "nombre_de_tu_modelo"
  
  model_version:
    default: "Production"
  
  warehouse_id:
    default: "tu_warehouse_id"
```

## 📦 Despliegue

### 1. Asegurar que el modelo está registrado en MLflow

```python
import mlflow

# Registrar modelo
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="my_ml_model"
)

# Promover a Production
client = mlflow.tracking.MlflowClient()
client.set_registered_model_alias(
    name="my_ml_model",
    alias="Production",
    version=1
)
```

### 2. Incluir en databricks.yml

Agrega en el archivo principal `databricks.yml`:

```yaml
include:
  - resources/*.yml
  - flask_app/app.yml
```

### 3. Desplegar con DAB

```bash
# Validar configuración
databricks bundle validate --target dev

# Desplegar
databricks bundle deploy --target dev

# Ejecutar app
databricks bundle run ml_flask_app --target dev
```

## 🧪 Desarrollo Local

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Configurar variables de entorno

```bash
export MODEL_NAME="my_ml_model"
export MODEL_VERSION="Production"
```

### 3. Ejecutar aplicación

```bash
python app.py
```

La aplicación estará disponible en `http://localhost:8000`

## 🔧 Personalización

### Modificar el modelo de respuesta

Edita la función `predict()` en `app.py` para personalizar el formato de respuesta:

```python
@app.route("/predict", methods=["POST"])
def predict():
    # Tu lógica personalizada
    pass
```

### Agregar nuevos endpoints

```python
@app.route("/custom-endpoint", methods=["GET"])
def custom_endpoint():
    return jsonify({"message": "Custom response"})
```

### Modificar la interfaz web

* Edita `templates/index.html` para cambiar el contenido
* Modifica `static/style.css` para ajustar estilos

## 📊 Integración con el Pipeline ML

Esta aplicación se integra con el pipeline de ML en la carpeta `ml_models/`:

1. El pipeline entrena y registra modelos en MLflow
2. La app Flask carga automáticamente el modelo desde MLflow
3. Los endpoints exponen predicciones a través de API REST

## 🔒 Seguridad

* Usa secretos de Databricks para tokens sensibles
* Implementa autenticación en endpoints si es necesario
* Valida y sanitiza inputs antes de hacer predicciones
* Considera rate limiting para producción

## 📝 Notas

* La aplicación usa gunicorn en producción (configurado en requirements.txt)
* Los logs se pueden ver en la consola de Databricks Apps
* Para debugging, revisa los logs del servicio en Databricks
