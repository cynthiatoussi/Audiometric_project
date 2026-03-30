"""
Node de data engineering pour l'audiométrie vocale.

Même logique que le tonal, adapté aux scores WRS :
- Scores en pourcentage d'intelligibilité (0 à 100%)
- 7 intensités de 30 à 90 dB
- 14 colonnes : 7 WRS before + 7 WRS after

Trois étapes :
1. Vérification des colonnes attendues
2. Conversion en numérique
3. Suppression des lignes incohérentes
"""

import pandas as pd
import logging

log = logging.getLogger(__name__)

INTENSITIES = [30, 40, 50, 60, 70, 80, 90]

BEFORE_COLUMNS = [f"wrs_{i}_before" for i in INTENSITIES]
AFTER_COLUMNS = [f"wrs_{i}_after" for i in INTENSITIES]
REQUIRED_COLUMNS = BEFORE_COLUMNS + AFTER_COLUMNS

# Bornes : WRS est un pourcentage
MIN_WRS = 0
MAX_WRS = 100


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie que les 14 colonnes (7 WRS before + 7 WRS after) sont présentes.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    log.info(
        "Validation vocale OK : %d lignes, %d colonnes",
        len(df), len(df.columns)
    )

    return df[REQUIRED_COLUMNS]


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit toutes les colonnes en numérique.
    Les lettres et valeurs invalides deviennent NaN.
    """
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    n_converted = df_numeric.isna().sum().sum() - df.isna().sum().sum()
    if n_converted > 0:
        log.warning("%d valeurs non numériques converties en NaN", n_converted)

    return df_numeric


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes contenant :
    - des NaN
    - des WRS < 0%
    - des WRS > 100%
    """
    n_before = len(df)

    invalid_mask = (
        df.isna()
        | (df < MIN_WRS)
        | (df > MAX_WRS)
    ).any(axis=1)

    cleaned_df = df[~invalid_mask]

    n_removed = n_before - len(cleaned_df)
    log.info(
        "Nettoyage vocal : %d lignes supprimées sur %d (%.1f%%)",
        n_removed, n_before, n_removed / n_before * 100
    )

    return cleaned_df