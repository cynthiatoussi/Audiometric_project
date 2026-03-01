"""
Nodes du pipeline inference.

Ce pipeline :
- valide les données
- détecte les lignes invalides
- prédit les seuils AFTER
"""

import pandas as pd


BEFORE_COLUMNS = [
    "before_exam_125_Hz",
    "before_exam_250_Hz",
    "before_exam_500_Hz",
    "before_exam_1000_Hz",
    "before_exam_2000_Hz",
    "before_exam_4000_Hz",
    "before_exam_8000_Hz",
]


def validate_input(df: pd.DataFrame):
    """
    Vérifie la présence des colonnes attendues.
    """
    missing = set(BEFORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    return df


def convert_numeric(df: pd.DataFrame):
    """
    Convertit en numérique.
    """
    df_numeric = df.copy()
    for col in BEFORE_COLUMNS:
        df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")
    return df_numeric


def detect_invalid_rows(df: pd.DataFrame):
    """
    Détecte les lignes invalides.
    """
    invalid_mask = (
        df.isna() |
        (df < 0) |
        (df > 120)
    ).any(axis=1)

    invalid_indices = df[invalid_mask].index.tolist()
    return invalid_indices


def predict(model, df: pd.DataFrame):
    """
    Effectue la prédiction sur les lignes valides.
    """
    predictions = model.predict(df)
    return predictions