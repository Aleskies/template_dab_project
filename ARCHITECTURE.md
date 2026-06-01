# 🏗️ Arquitectura del Proyecto ML Template

## 📊 Visión General

Este proyecto implementa una arquitectura completa de Machine Learning en Databricks que incluye:

* **Pipeline de ML** end-to-end (data → training → model registry)
* **Aplicación Flask** para servir modelos vía API REST
* **Automatización con DABs** para deployment y scheduling
* **Integración MLflow** para tracking y model management

## 🎯 Arquitectura de Componentes

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATABRICKS WORKSPACE                          │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    UNITY CATALOG                            │ │
│  │                                                             │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │ │
│  │  │  Raw Data    │  │  Features    │  │ Predictions  │    │ │
│  │  │   Tables     │  │   Tables     │  │   Tables     │    │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘    │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    ML PIPELINE                              │ │
│  │                                                             │ │
│  │  ┌───────────────┐      ┌───────────────┐                 │ │
│  │  │   Feature     │──────▶│   Training    │                 │ │
│  │  │ Engineering   │      │    Script     │                 │ │
│  │  └───────────────┘      └───────────────┘                 │ │
│  │         │                       │                          │ │
│  │         │                       ▼                          │ │
│  │         │              ┌───────────────┐                  │ │
│  │         │              │    MLflow     │                  │ │
│  │         │              │  Experiment   │                  │ │
│  │         │              │   Tracking    │                  │ │
│  │         │              └───────────────┘                  │ │
│  │         │                       │                          │ │
│  │         │                       ▼                          │ │
│  │         │              ┌───────────────┐                  │ │
│  │         └─────────────▶│     Model     │                  │ │
│  │                        │   Registry    │                  │ │
│  │                        └───────────────┘                  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   FLASK APPLICATION                         │ │
│  │                                                             │ │
│  │  ┌───────────────┐      ┌───────────────┐                 │ │
│  │  │  Web UI       │      │   REST API    │                 │ │
│  │  │  (HTML/CSS)   │      │  /predict     │                 │ │
│  │  └───────────────┘      └───────────────┘                 │ │
│  │         │                       │                          │ │
│  │         └───────────────────────┘                          │ │
│  │                    │                                        │ │
│  │                    ▼                                        │ │
│  │           ┌───────────────┐                                │ │
│  │           │ Loaded Model  │                                │ │
│  │           │  from MLflow  │                                │ │
│  │           └───────────────┘                                │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │              DATABRICKS JOBS (DABs)                         │ │
│  │                                                             │ │
│  │  ┌───────────────┐      ┌───────────────┐                 │ │
│  │  │  Training Job │      │ Prediction Job│                 │ │
│  │  │   (Scheduled) │      │  (On-demand)  │                 │ │
│  │  └───────────────┘      └───────────────┘                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## 📂 Estructura de Archivos

```
template_ml_dab_project/
│
├── 📄 databricks.yml              # Bundle principal
├── 📄 pyproject.toml              # Dependencias
├── 📄 README.md                   # Documentación principal
├── 📄 QUICKSTART.md               # Guía rápida
├── 📄 ARCHITECTURE.md             # Este archivo
├── 📄 create_sample_data.py       # Utilidad para datos de prueba
│
├── 📁 ml_models/                  # 🤖 Módulo de ML
│   ├── __init__.py
│   ├── train.py                   # ⚙️ Entrenamiento
│   ├── predict.py                 # 🔮 Predicción
│   ├── feature_engineering.py     # 🛠️ Feature engineering
│   ├── model_config.py            # ⚙️ Configuración
│   ├── mlflow_utils.py            # 📊 MLflow tracking
│   └── README.md                  # Documentación ML
│
├── 📁 flask_app/                  # 🌐 Aplicación Web
│   ├── app.py                     # 🚀 App principal
│   ├── requirements.txt           # 📦 Dependencias Flask
│   ├── app.yml                    # ⚙️ Config Databricks App
│   ├── README.md                  # Documentación Flask
│   ├── templates/
│   │   └── index.html             # 🎨 UI principal
│   └── static/
│       └── style.css              # 🎨 Estilos
│
├── 📁 resources/                  # 🔧 Definiciones DABs
│   ├── sample_job.job.yml
│   └── ml_training.job.yml        # Jobs de ML
│
├── 📁 src/                        # Código fuente del paquete
├── 📁 tests/                      # Tests unitarios
└── 📁 fixtures/                   # Datos de prueba
```

## 🔄 Flujo de Datos

### 1️⃣ Fase de Entrenamiento

```
Unity Catalog Tables
        │
        ▼
Feature Engineering
        │
        ▼
   Model Training
        │
        ├──▶ MLflow Tracking (metrics, params)
        │
        ▼
  Model Registry
        │
        └──▶ Alias: Production/Staging
```

### 2️⃣ Fase de Predicción (Batch)

```
Unity Catalog Input Table
        │
        ▼
  Load Model (from Registry)
        │
        ▼
Feature Engineering
        │
        ▼
 Batch Predictions
        │
        ▼
Unity Catalog Output Table
```

### 3️⃣ Fase de Servicio (Flask App)

```
HTTP Request (JSON)
        │
        ▼
   Flask Endpoint
        │
        ▼
  Load Model (cached)
        │
        ▼
Feature Engineering
        │
        ▼
    Prediction
        │
        ▼
HTTP Response (JSON)
```

## 🔌 Puntos de Integración

### 1. Unity Catalog
* **Tablas de entrada**: `catalog.schema.train_data`, `test_data`
* **Tablas de salida**: `catalog.schema.predictions`
* **Gobernanza**: Permisos y lineage automático

### 2. MLflow
* **Experiments**: `/Shared/ml_experiments`
* **Model Registry**: Modelos versionados con aliases
* **Tracking**: Métricas, parámetros, artefactos

### 3. Databricks Jobs
* **Training Job**: Entrena y registra modelos
* **Prediction Job**: Batch scoring
* **Scheduling**: Cron expressions para automatización

### 4. Databricks Apps
* **Flask App**: Serve modelo vía REST API
* **Compute**: SQL Warehouse serverless
* **Endpoints**: `/predict`, `/health`, `/model-info`

## 🛠️ Tecnologías Utilizadas

### Backend ML
* **Python 3.10+**
* **scikit-learn**: Modelos base
* **pandas/numpy**: Manipulación de datos
* **MLflow**: Experiment tracking
* **PySpark**: Procesamiento distribuido

### Web Application
* **Flask**: Framework web
* **HTML/CSS**: Frontend
* **JavaScript**: Interactividad
* **Gunicorn**: Production server

### DevOps
* **Databricks Asset Bundles (DABs)**: IaC
* **Unity Catalog**: Data governance
* **Git**: Version control

## 🔐 Seguridad y Gobernanza

### Unity Catalog
* Control de acceso granular (GRANT/REVOKE)
* Lineage automático de datos
* Auditoría completa

### Model Registry
* Versionado de modelos
* Transiciones de stage auditadas
* Tags y documentación

### Flask App
* Variables de entorno para secretos
* Validación de inputs
* Error handling robusto

## 📈 Escalabilidad

### Compute
* **Jobs**: Clusters auto-scaling
* **Flask App**: SQL Warehouse serverless
* **Training**: Distributed con Spark

### Data
* **Unity Catalog**: Petabytes de datos
* **Delta Lake**: ACID transactions
* **Partitioning**: Optimización automática

## 🔄 CI/CD Pipeline

```
Development
    │
    ▼
databricks bundle validate
    │
    ▼
databricks bundle deploy --target dev
    │
    ▼
Run tests & validation
    │
    ▼
databricks bundle deploy --target prod
    │
    ▼
Production Deployment
```

## 📊 Monitoreo y Observabilidad

### MLflow UI
* Comparación de runs
* Visualización de métricas
* Model performance tracking

### Databricks Jobs UI
* Job execution history
* Error logs
* Performance metrics

### Flask App Logs
* Request/response logs
* Error tracking
* Health monitoring

## 🎯 Best Practices Implementadas

✅ **Separation of Concerns**: ML logic, API, configuración separados  
✅ **Configuration as Code**: Todo en YAML/Python  
✅ **Version Control**: Modelos y código versionados  
✅ **Reproducibility**: Seeds fijos, tracking completo  
✅ **Modularity**: Componentes reutilizables  
✅ **Documentation**: READMEs en cada módulo  
✅ **Testing**: Estructura para tests unitarios  
✅ **Security**: Uso de secretos y variables de entorno  

## 🚀 Deployment Targets

### Dev Environment
* Modo: `development`
* Recursos: Single node
* Schedule: Deshabilitado
* Propósito: Testing y desarrollo

### Prod Environment
* Modo: `production`
* Recursos: Multi-node clusters
* Schedule: Habilitado
* Propósito: Producción real

## 📞 Extensibilidad

### Agregar Nuevos Modelos
1. Editar `model_config.py`
2. Implementar en `train.py`
3. Actualizar `predict.py` si necesario

### Agregar Nuevos Endpoints
1. Editar `flask_app/app.py`
2. Actualizar templates si necesario
3. Documentar en README

### Agregar Nuevos Jobs
1. Crear archivo `.job.yml` en `resources/`
2. Agregar al `include` en `databricks.yml`
3. Desplegar con `bundle deploy`

## 💡 Mejoras Futuras

* [ ] Integración con Feature Store
* [ ] A/B testing de modelos
* [ ] Automated retraining triggers
* [ ] Advanced monitoring (Prometheus/Grafana)
* [ ] CI/CD con GitHub Actions
* [ ] Model drift detection
* [ ] Explainability (SHAP/LIME)
* [ ] Multi-model serving
* [ ] Canary deployments

## 📚 Referencias

* [Databricks ML Guide](https://docs.databricks.com/machine-learning/index.html)
* [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
* [Unity Catalog](https://docs.databricks.com/data-governance/unity-catalog/index.html)
* [Databricks Asset Bundles](https://docs.databricks.com/dev-tools/bundles/index.html)
* [Flask Documentation](https://flask.palletsprojects.com/)
