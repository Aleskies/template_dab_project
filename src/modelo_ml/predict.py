"""Script de ejemplo para realizar predicciones usando el modelo guardado"""
from src.models.predict import predict
import pandas as pd


def main():
    df = pd.read_csv('data/processed/new_data.csv')
    preds = predict(df, model_path='models/model_final.pkl')
    print(preds.head())


if __name__ == '__main__':
    main()
