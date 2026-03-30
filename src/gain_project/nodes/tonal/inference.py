"""
Node d'inférence pour l'audiométrie tonale.

Ce module :
- Valide les données d'entrée (colonnes before uniquement)
- Convertit en numérique
- Nettoie les lignes invalides
- Applique le même feature engineering que le training (PTA + pente)
- Prédit les gains prothétiques

IMPORTANT : les bornes et les features doivent rester
synchronisées avec preprocess.py et split.py
"""



import pandas as pd
import logging

log = logging.getLogger(__name__)

FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000]
BEFORE_COLUMNS = [f"before_exam_{f}_Hz" for f in FREQUENCIES]

# Même borne que preprocess.py
MAX_DB = 120


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie que les 7 colonnes before sont présentes.
    En inférence on ne reçoit que les before, pas les after.
    """
    missing = set(BEFORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    log.info("Validation inférence OK : %d lignes", len(df))
    return df[BEFORE_COLUMNS]


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit en numérique.
    Même logique que dans preprocess.py.
    """
    return df.apply(pd.to_numeric, errors="coerce")


def clean_input(df: pd.DataFrame):
    """
    Supprime les lignes invalides.
    Mêmes critères que preprocess.py : NaN ou > 120 dB.

    Retourne :
    - cleaned : DataFrame nettoyé
    - invalid_report : dict avec le nombre et indices des lignes rejetées
    """
    n_before = len(df)

    invalid_mask = (
        df.isna() | (df > MAX_DB)
    ).any(axis=1)

    cleaned = df[~invalid_mask].reset_index(drop=True)
    invalid_indices = df[invalid_mask].index.tolist()
    n_removed = n_before - len(cleaned)

    if n_removed > 0:
        log.warning(
            "Inférence : %d lignes invalides retirées sur %d",
            n_removed, n_before
        )

    invalid_report = {
        "count": n_removed,
        "total": n_before,
        "indices": invalid_indices,
    }

    return cleaned, invalid_report


def predict(model, df: pd.DataFrame) -> pd.DataFrame:
    """
    Effectue la prédiction directement sur les 7 colonnes before.
    """
    after_columns = [f"after_exam_{f}_Hz" for f in FREQUENCIES]

    predictions = model.predict(df)
    result = pd.DataFrame(predictions, columns=after_columns)

    log.info("Prédictions : %d lignes", len(result))
    return result