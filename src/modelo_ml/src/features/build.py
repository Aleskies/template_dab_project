import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    # Ejemplo: crear features sencillas
    if 'price' in df.columns and 'quantity' in df.columns:
        df['total'] = df['price'] * df['quantity']
    return df
