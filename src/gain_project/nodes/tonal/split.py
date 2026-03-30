"""
Node de feature engineering pour l'audiométrie tonale.

Ce module :
- Calcule des features dérivées audiologiques
- Sépare les features (X) de la cible (y)

Features ajoutées :
- PTA (Pure Tone Average) : moyenne des seuils à 500, 1000, 2000 Hz
  → indicateur standard de sévérité de la perte auditive
- Pente : différence entre hautes et basses fréquences
  → permet de distinguer les profils slope / reverse / plat
"""


import pandas as pd
import logging

log = logging.getLogger(__name__)

FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000]
BEFORE_COLUMNS = [f"before_exam_{f}_Hz" for f in FREQUENCIES]
AFTER_COLUMNS = [f"after_exam_{f}_Hz" for f in FREQUENCIES]


def split_features_targets(df: pd.DataFrame):
    """
    Sépare le DataFrame nettoyé en :

    X : 7 colonnes before (seuils auditifs par fréquence)
    y : 7 colonnes after (cible à prédire)
    """
    X = df[BEFORE_COLUMNS]
    y = df[AFTER_COLUMNS]

    log.info(
        "Feature engineering : X = %d features, y = %d cibles, %d échantillons",
        X.shape[1], y.shape[1], len(X)
    )

    return X, y