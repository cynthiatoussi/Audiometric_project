"""
Node d'inférence pour l'audiométrie vocale.

Ce module :
- Valide les données d'entrée (7 colonnes WRS before)
- Convertit en numérique
- Nettoie les lignes invalides
- Prédit les WRS after

IMPORTANT : les bornes doivent rester synchronisées avec preprocess.py
"""

import pandas as pd
import logging

log = logging.getLogger(__name__)

INTENSITIES = [30, 40, 50, 60, 70, 80, 90]
BEFORE_COLUMNS = [f"wrs_{i}_before" for i in INTENSITIES]

# Mêmes bornes que preprocess.py
MIN_WRS = 0
MAX_WRS = 100


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie que les 7 colonnes WRS before sont présentes.
    En inférence on ne reçoit que les before.
    """
    missing = set(BEFORE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    log.info("Validation inférence vocale : %d lignes", len(df))
    return df[BEFORE_COLUMNS]


def convert_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit en numérique.
    Même logique que preprocess.py.
    """
    return df.apply(pd.to_numeric, errors="coerce")


def clean_input(df: pd.DataFrame):
    """
    Supprime les lignes invalides.
    Mêmes critères que preprocess.py : NaN, < 0% ou > 100%.

    Retourne :
    - cleaned : DataFrame nettoyé
    - invalid_report : dict avec le nombre et indices des lignes rejetées
    """
    n_before = len(df)

    invalid_mask = (
        df.isna()
        | (df < MIN_WRS)
        | (df > MAX_WRS)
    ).any(axis=1)

    cleaned = df[~invalid_mask].reset_index(drop=True)
    invalid_indices = df[invalid_mask].index.tolist()
    n_removed = n_before - len(cleaned)

    if n_removed > 0:
        log.warning(
            "Inférence vocale : %d lignes invalides retirées sur %d",
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
    Prédit les WRS after.

    Pas de feature engineering supplémentaire pour le vocal
    (contrairement au tonal qui ajoute PTA + pente).
    Les 7 WRS before décrivent déjà la courbe complète.
    """
    after_columns = [f"wrs_{i}_after" for i in INTENSITIES]

    predictions = model.predict(df)
    result = pd.DataFrame(predictions, columns=after_columns)

    log.info("Prédictions vocales : %d lignes", len(result))
    return result