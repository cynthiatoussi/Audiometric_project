"""
Nodes du pipeline training.

Ce pipeline :
1. Entraîne un modèle
2. Évalue les performances
"""

import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score



def evaluate_model(model, X_test, y_test):
    """
    Calcule les métriques de performance.
    """
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "mse": float(mse),
        "r2": float(r2)
    }

    print("Model performance:", metrics)

    return metrics