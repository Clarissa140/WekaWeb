import pandas as pd

def load_dataset(filepath):
    df = pd.read_csv(filepath)
    X = df.iloc[:, :-1]  # todas las columnas menos la última
    y = df.iloc[:, -1]   # última columna como target
    return X, y
