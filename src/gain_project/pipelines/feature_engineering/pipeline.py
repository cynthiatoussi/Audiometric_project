"""
Pipeline feature_engineering.

Ce pipeline :
- Prend les données nettoyées
- Sépare X et y
- Applique un train/test split
"""

from kedro.pipeline import Pipeline, node
from gain_project.nodes.split import (
    separate_features_target,
    split_train_test,
)

def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline([

        node(
            func=separate_features_target,
            inputs="clean_tonal_data",
            outputs=["X_full", "y_full"],
            name="separate_features_target_node",
        ),

        node(
            func=split_train_test,
            inputs=["X_full", "y_full"],
            outputs=["X_train", "X_test", "y_train", "y_test"],
            name="train_test_split_node",
        ),
    ])