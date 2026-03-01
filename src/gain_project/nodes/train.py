"""
Nodes du pipeline training.

Ce pipeline :
1. Entraîne un modèle
2. Évalue les performances
"""

import json
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def train_model(X_train, y_train):
    """
    Entraîne un RandomForest multi-output.
    """
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model

