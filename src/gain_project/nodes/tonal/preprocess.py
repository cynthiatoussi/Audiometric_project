"""
Node de data engineering pour l'audiométrie tonale.

Ce module gère le nettoyage des données brutes générées
par le simulateur du professeur, qui injecte :
- des valeurs NaN
- des lettres aléatoires
- des floats inattendus
- des outliers (> 120 dB)

Trois étapes :
1. Vérification des colonnes attendues
2. Conversion des valeurs en numérique
3. Suppression des lignes incohérentes
"""

import pandas as pd
import logging

log = logging.getLogger(__name__)

# Fréquences standard de l'audiogramme tonal
FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000]

# Colonnes attendues dans le dataset
BEFORE_COLUMNS = [f"before_exam_{f}_Hz" for f in FREQUENCIES]
AFTER_COLUMNS = [f"after_exam_{f}_Hz" for f in FREQUENCIES]
REQUIRED_COLUMNS = BEFORE_COLUMNS + AFTER_COLUMNS

# Borne max physiologique
# Les valeurs négatives (-10 à 0) sont valides en audiologie
# Le générateur injecte des outliers entre 120 et 150 qu'il faut retirer
MAX_DB = 120


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vérifie que toutes les colonnes nécessaires sont présentes.

    Le générateur produit exactement 14 colonnes :
    7 before + 7 after pour les fréquences 125 à 8000 Hz.

    Si une colonne manque, le pipeline s'arrête immédiatement.
    """
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    log.info(
        "Validation OK : %d lignes, %d colonnes",
        len(df), len(df.columns)
    )

    return df[REQUIRED_COLUMNS]


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convertit toutes les colonnes en numérique.

    Le générateur injecte des lettres aléatoires (A-Z)
    et des floats là où on attend des entiers.
    Tout ce qui n'est pas convertible devient NaN.
    """
    df_numeric = df.apply(pd.to_numeric, errors="coerce")

    n_converted = df_numeric.isna().sum().sum() - df.isna().sum().sum()
    if n_converted > 0:
        log.warning("%d valeurs non numériques converties en NaN", n_converted)

    return df_numeric


def remove_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Supprime les lignes contenant :
    - des valeurs NaN (issues de la conversion ou injectées)
    - des valeurs > 120 dB (outliers injectés par le générateur)
    """
    n_before = len(df)

    invalid_mask = (
        df.isna()
        | (df > MAX_DB)
    ).any(axis=1)

    cleaned_df = df[~invalid_mask]

    n_removed = n_before - len(cleaned_df)
    log.info(
        "Nettoyage : %d lignes supprimées sur %d (%.1f%%)",
        n_removed, n_before, n_removed / n_before * 100
    )

    return cleaned_df