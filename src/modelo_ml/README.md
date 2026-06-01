# modelo_ml

Estructura simplificada para análisis y modelos.

Estructura:

modelo_ml/
├── data/ (raw, processed, external)
├── notebooks/ (01_eda.ipynb, 02_feature_eng.ipynb, 03_modeling.ipynb)
├── src/ (código reutilizable: data, features, models, utils)
├── models/ (modelos guardados)
├── config/ (params.yaml)
├── requirements.txt
├── main.py
└── predict.py

Usa esta carpeta como un workspace independiente para desarrollo y notebooks.

CLI local (env activado):

```powershell
# Ejecuta el trainer incorporado (usa el ejemplo simulado en package)
python -m template_ml_dab_project.ml --help
```
