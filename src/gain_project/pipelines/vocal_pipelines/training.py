"""
Pipeline d'entraînement vocal.

1. Split train/test (80/20)
2. Entraînement Random Forest + Optuna + MLflow
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.vocal.train import split_train_test, train_rf_optuna


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=split_train_test,
            inputs=["X_vocal", "y_vocal"],
            outputs=["X_train_vocal", "X_test_vocal",
                     "y_train_vocal", "y_test_vocal"],
            name="vocal_split_train_test_node",
        ),

        node(
            func=train_rf_optuna,
            inputs=["X_train_vocal", "y_train_vocal",
                    "X_test_vocal", "y_test_vocal"],
            outputs="rf_vocal_model",
            name="vocal_train_rf_node",
        ),
    ])