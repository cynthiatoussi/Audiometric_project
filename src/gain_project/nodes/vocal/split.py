"""
Node de feature engineering pour l'audiométrie vocale.

Sépare les features (WRS before) de la cible (WRS after).


ici les 7 scores WRS à différentes intensités décrivent
déjà la forme complète de la courbe sigmoïde :
- la montée (pente)
- le plateau (Wmax)
- le rollover éventuel (chute à 80-90 dB)

Pas besoin de features dérivées supplémentaires.
"""

import pandas as pd
import logging

log = logging.getLogger(__name__)

INTENSITIES = [30, 40, 50, 60, 70, 80, 90]
BEFORE_COLUMNS = [f"wrs_{i}_before" for i in INTENSITIES]
AFTER_COLUMNS = [f"wrs_{i}_after" for i in INTENSITIES]


def split_features_targets(df: pd.DataFrame):
    """
    Sépare le DataFrame en :

    X : 7 colonnes WRS before (% intelligibilité à 30-90 dB)
    y : 7 colonnes WRS after (% intelligibilité à 30-90 dB)
    """
    X = df[BEFORE_COLUMNS]
    y = df[AFTER_COLUMNS]

    log.info(
        "Feature engineering vocal : X = %d features, y = %d cibles, %d échantillons",
        X.shape[1], y.shape[1], len(X)
    )

    return X, y