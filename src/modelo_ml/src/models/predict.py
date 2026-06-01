import joblib
import pandas as pd


def predict(inputs: pd.DataFrame, model_path: str = 'models/model_final.pkl') -> pd.DataFrame:
    model = joblib.load(model_path)
    preds = model.predict(inputs)
    return pd.DataFrame({'prediction': preds})
