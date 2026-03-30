"""
Générateur d'audiométrie vocale.

Simule des courbes d'intelligibilité (WRS) réalistes
basées sur une fonction sigmoïde.

5 profils cliniques tirés de l'audiogramme vocal de référence :

1. Normal (courbe verte)
   SRT bas (~15 dB), pente raide, plateau à ~100%

2. Surdité de Transmission (courbe bleue)
   Courbe décalée à droite, même forme que normal, plateau ~100%
   La prothèse compense bien : bonne amélioration

3. Surdité de Perception (courbe rouge)
   Décalée à droite, pente plus douce, plateau réduit (~70-90%)
   La prothèse aide partiellement

4. Perception + Recrutement (courbe violette)
   Pente raide (recrutement = sensibilité anormale au volume)
   Plateau réduit (~65-85%)

5. Perception + Surrecrutement (courbe avec rollover)
   Chute d'intelligibilité aux fortes intensités (≥ 80 dB)
   La prothèse atténue le rollover mais ne l'élimine pas

Modèle mathématique :
    WRS(I) = Wmax / (1 + exp(-k * (I - SRT)))
    avec rollover optionnel aux hautes intensités

Intensités mesurées : 30, 40, 50, 60, 70, 80, 90 dB
Scores : pourcentage d'intelligibilité (0 à 100%)

Injection de bruit identique au générateur tonal :
- floats aléatoires
- valeurs NaN
- lettres aléatoires
"""

import pandas as pd
import numpy as np
import random
import logging

log = logging.getLogger(__name__)

INTENSITIES = [30, 40, 50, 60, 70, 80, 90]


# -------------------------------------------------
# Courbe sigmoïde d'intelligibilité
# -------------------------------------------------

def logistic_curve(intensity, srt, slope, wmax):
    """
    Modèle logistique de la courbe d'intelligibilité vocale.

    Paramètres :
        intensity : intensité sonore en dB
        srt : Speech Reception Threshold (intensité à 50% d'intelligibilité)
        slope : pente de la courbe (raideur de la montée)
        wmax : plateau maximum (% d'intelligibilité max atteignable)

    Retourne :
        score WRS en %
    """
    return wmax / (1 + np.exp(-slope * (intensity - srt)))


# -------------------------------------------------
# Paramètres par profil clinique
# -------------------------------------------------

def generate_profile_params(profile):
    """
    Génère les paramètres de la courbe sigmoïde selon le profil.
    Chaque profil correspond à une pathologie réelle.

    Retourne : (srt, slope, wmax, rollover)
    """

    if profile == "normal":
        # Courbe 1 (verte) : SRT bas, pente raide, plateau 100%
        return (
            random.uniform(10, 25),       # SRT
            random.uniform(0.18, 0.30),   # pente raide
            random.uniform(95, 100),      # plateau élevé
            0,                            # pas de rollover
        )

    elif profile == "transmission":
        # Courbe 2 (bleue) : décalée à droite, plateau conservé
        return (
            random.uniform(35, 55),
            random.uniform(0.15, 0.25),
            random.uniform(90, 100),
            0,
        )

    elif profile == "perception":
        # Courbe 3 (rouge) : décalée, pente douce, plateau réduit
        return (
            random.uniform(40, 65),
            random.uniform(0.08, 0.15),
            random.uniform(70, 90),
            0,
        )

    elif profile == "perception_recrutement":
        # Courbe 4 (violette) : pente raide (recrutement), plateau réduit
        return (
            random.uniform(45, 65),
            random.uniform(0.15, 0.25),
            random.uniform(65, 85),
            0,
        )

    elif profile == "perception_surrecrutement":
        # Courbe 5 : rollover aux fortes intensités
        return (
            random.uniform(40, 60),
            random.uniform(0.10, 0.20),
            random.uniform(75, 95),
            random.uniform(10, 35),  # rollover
        )


# -------------------------------------------------
# Scores BEFORE (sans prothèse)
# -------------------------------------------------

def generate_before_scores(srt, slope, wmax, rollover):
    """
    Génère les WRS avant prothèse pour chaque intensité.
    Applique le rollover si surrecrutement.
    Ajoute du bruit de mesure clinique.
    """
    scores = []
    for intensity in INTENSITIES:
        score = logistic_curve(intensity, srt, slope, wmax)

        # Rollover : chute progressive au-delà de 80 dB
        if intensity >= 80 and rollover > 0:
            drop = rollover * (intensity - 75) / 15
            score -= drop

        # Bruit clinique (variabilité de mesure réelle)
        score += np.random.normal(0, 3)
        score = np.clip(score, 0, 100)
        scores.append(round(score, 1))

    return scores


# -------------------------------------------------
# Scores AFTER (avec prothèse)
# -------------------------------------------------

def calculate_after_scores(srt, slope, wmax, rollover):
    """
    Génère les WRS après prothèse.
    Amélioration simulée :
    - SRT abaissé (décalage de la courbe vers la gauche)
    - Pente légèrement améliorée
    - Rollover atténué mais pas supprimé
    """
    improved_srt = srt - random.uniform(3, 12)
    improved_slope = slope + random.uniform(0.01, 0.05)

    scores = []
    for intensity in INTENSITIES:
        score = logistic_curve(intensity, improved_srt, improved_slope, wmax)

        # Rollover atténué par la prothèse
        if intensity >= 80 and rollover > 0:
            drop = rollover * random.uniform(0.2, 0.6) * (intensity - 75) / 15
            score -= drop

        score += np.random.normal(0, 3)
        score = np.clip(score, 0, 100)
        scores.append(round(score, 1))

    return scores


# -------------------------------------------------
# Injection de bruit (même logique que tonal)
# -------------------------------------------------

def add_realism(df):
    """
    Injecte du bruit réaliste dans le dataset :
    - floats aléatoires
    - valeurs NaN
    - lettres aléatoires
    Modifie entre 20% et 40% des lignes.
    """
    df = df.copy()

    for col in df.columns:
        df[col] = df[col].astype("object")

    num_rows = len(df)
    num_changes = random.randint(int(num_rows * 0.2), int(num_rows * 0.4))

    for _ in range(num_changes):
        row = random.randint(0, num_rows - 1)
        col = random.choice(df.columns)
        mod = random.choice(["float", "NaN", "letter"])

        if mod == "float":
            df.at[row, col] = float(np.round(random.uniform(0, 100), 2))
        elif mod == "NaN":
            df.at[row, col] = np.nan
        elif mod == "letter":
            df.at[row, col] = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    return df


# -------------------------------------------------
# Fonction principale appelée par Kedro
# -------------------------------------------------

def generate_vocal_dataset(n_samples: int) -> pd.DataFrame:
    """
    Génère n_samples examens vocaux puis injecte du bruit.

    Paramètre :
        n_samples : nombre d'examens (via parameters.yml)

    Retourne :
        DataFrame avec 14 colonnes (7 WRS before + 7 WRS after) bruité
    """
    log.info("Génération de %d examens vocaux...", n_samples)

    profiles = [
        "normal",
        "transmission",
        "perception",
        "perception_recrutement",
        "perception_surrecrutement",
    ]

    headers_before = [f"wrs_{i}_before" for i in INTENSITIES]
    headers_after = [f"wrs_{i}_after" for i in INTENSITIES]
    columns = headers_before + headers_after

    data = []

    for _ in range(n_samples):
        profile = random.choice(profiles)
        srt, slope, wmax, rollover = generate_profile_params(profile)

        before = generate_before_scores(srt, slope, wmax, rollover)
        after = calculate_after_scores(srt, slope, wmax, rollover)

        data.append(before + after)

    df = pd.DataFrame(data, columns=columns)
    df = add_realism(df)

    log.info("Dataset vocal généré : %d lignes, %d colonnes", len(df), len(df.columns))

    return df