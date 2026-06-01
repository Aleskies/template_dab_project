import pandas as pd


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    # Ejemplo simple: eliminar duplicados y filas con NA completas
    df = df.drop_duplicates()
    df = df.dropna(how='all')
    return df
