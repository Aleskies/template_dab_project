"""Script de ejemplo para entrenar modelo completo"""
from src.data.load import load_csv
from src.data.clean import clean_df
from src.features.build import build_features
from src.models.train import train_model


def main():
    df = load_csv('data/processed/train.csv')
    df = clean_df(df)
    df = build_features(df)
    X = df.drop(columns=['target'])
    y = df['target']
    train_model(X, y, save_path='models/model_final.pkl')


if __name__ == '__main__':
    main()
