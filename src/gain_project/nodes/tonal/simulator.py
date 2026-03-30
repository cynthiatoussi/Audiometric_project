"""
Node de génération des données d'audiométrie tonale.

Code du générateur fourni par le professeur, adapté comme node Kedro.
Génère des audiogrammes réalistes avec :
- 7 profils auditifs (normal, mild, moderate, severe, profound, slope, reverse)
- Injection d'outliers (10% des cas)
- Injection de bruit réaliste (floats, NaN, lettres)
"""

import pandas as pd
import numpy as np
import random
import logging

log = logging.getLogger(__name__)


def generate_thresholds_by_profile(profile, frequencies):
    """Génère les seuils before selon le profil auditif."""

    if profile == "normal":
        return [random.randint(-10, 20) for _ in frequencies]
    elif profile == "mild":
        return [random.randint(20, 40) for _ in frequencies]
    elif profile == "moderate":
        return [random.randint(40, 60) for _ in frequencies]
    elif profile == "severe":
        return [random.randint(60, 80) for _ in frequencies]
    elif profile == "profound":
        return [random.randint(80, 120) for _ in frequencies]
    elif profile == "slope":
        return [random.randint(0, 25) + i * 10 for i, _ in enumerate(frequencies)]
    elif profile == "reverse":
        return [random.randint(70, 90) - i * 5 for i, _ in enumerate(frequencies)]
    else:
        return [random.randint(0, 100) for _ in frequencies]


def calculate_improvements(profile, thresholds):
    """Calcule les seuils after en fonction du profil."""

    if profile in ["normal", "mild"]:
        return np.clip(
            thresholds - np.random.randint(15, 30, size=len(thresholds)), 0, 120
        ).tolist()
    elif profile in ["moderate", "severe", "profound"]:
        return np.clip(
            thresholds - np.random.randint(5, 15, size=len(thresholds)), 0, 120
        ).tolist()
    elif profile == "slope":
        return np.clip(
            thresholds - np.random.randint(10, 20, size=len(thresholds)), 0, 120
        ).tolist()
    elif profile == "reverse":
        return np.clip(
            thresholds - np.random.randint(0, 10, size=len(thresholds)), 0, 120
        ).tolist()
    else:
        return thresholds


def generate_audiograms(exam_count, init_freq=125):
    """
    Génère les audiogrammes bruts.

    Paramètres :
        exam_count : nombre d'examens à générer
        init_freq : fréquence de départ (125 Hz par défaut)

    Retourne :
        DataFrame avec 14 colonnes (7 before + 7 after)
    """
    frequencies = [init_freq * (2 ** i) for i in range(7)]
    headers = [f"exam_{freq}_Hz" for freq in frequencies]
    headers_before = ["before_" + header for header in headers]
    headers_after = ["after_" + header for header in headers]
    columns = headers_before + headers_after

    data = []

    for _ in range(exam_count):
        profile = random.choice(
            ["normal", "mild", "moderate", "severe", "profound", "slope", "reverse"]
        )
        thresholds = generate_thresholds_by_profile(profile, frequencies)

        # Injection d'outliers (10% des cas)
        if random.random() < 0.1:
            outlier_position = random.randint(0, len(thresholds) - 1)
            if random.random() < 0.5:
                thresholds[outlier_position] = random.randint(120, 150)
            else:
                thresholds[outlier_position] = random.randint(-10, 0)

        improvements = calculate_improvements(profile, thresholds)
        data.append(thresholds + improvements)

    df = pd.DataFrame(data, columns=columns)
    return df


def add_realism_to_data(df):
    """
    Injecte du bruit réaliste dans le dataset :
    - floats aléatoires
    - valeurs NaN
    - lettres aléatoires
    Modifie entre 30% et 100% des lignes.
    """
    df = df.copy()

    for col in df.columns:
        df[col] = df[col].astype("object")

    num_rows, num_cols = df.shape
    num_changes = random.randint(int(np.round(num_rows * 0.3)), num_rows)

    for _ in range(num_changes):
        row_index = random.randint(0, num_rows - 1)
        col_index = random.choice(df.columns)
        modification_type = random.choice(["float", "NaN", "letter"])

        if modification_type == "float":
            df.at[row_index, col_index] = float(np.round(random.uniform(0, 100), 2))
        elif modification_type == "NaN":
            df.at[row_index, col_index] = np.nan
        elif modification_type == "letter":
            df.at[row_index, col_index] = str(
                random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            )

    return df


def generate_tonal_dataset(n_samples: int) -> pd.DataFrame:
    """
    Fonction principale appelée par Kedro.

    Génère n_samples audiogrammes puis injecte du bruit.

    Paramètre :
        n_samples : nombre d'audiogrammes (via parameters.yml)

    Retourne :
        DataFrame bruité prêt à être nettoyé par data_engineering
    """
    log.info("Génération de %d audiogrammes tonaux...", n_samples)

    df = generate_audiograms(n_samples)
    df = add_realism_to_data(df)

    log.info("Dataset généré : %d lignes, %d colonnes", len(df), len(df.columns))

    return df