"""
Nodes du pipeline feature_engineering.

Ce pipeline :
1. Sépare les variables X (before) et y (after)
2. Effectue un train/test split
"""

import pandas as pd
from sklearn.model_selection import train_test_split


BEFORE_COLUMNS = [
    "before_exam_125_Hz",
    "before_exam_250_Hz",
    "before_exam_500_Hz",
    "before_exam_1000_Hz",
    "before_exam_2000_Hz",
    "before_exam_4000_Hz",
    "before_exam_8000_Hz",
]

AFTER_COLUMNS = [
    "after_exam_125_Hz",
    "after_exam_250_Hz",
    "after_exam_500_Hz",
    "after_exam_1000_Hz",
    "after_exam_2000_Hz",
    "after_exam_4000_Hz",
    "after_exam_8000_Hz",
]


def separate_features_target(df: pd.DataFrame):
    """
    Sépare les variables d'entrée (X) et les variables cibles (y).
    """
    X = df[BEFORE_COLUMNS]
    y = df[AFTER_COLUMNS]
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.DataFrame):
    """
    Sépare les données en train et test.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test