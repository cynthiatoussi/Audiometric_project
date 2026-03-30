"""
Pipeline d'entraînement tonal.

1. Split train/test (80/20)
2. Entraînement XGBoost + MultiOutputRegressor + Optuna
3. Logging MLflow
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.tonal.train import split_train_test, train_xgb_optuna


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        # 1. Split train/test
        node(
            func=split_train_test,
            inputs=["X_tonal", "y_tonal"],
            outputs=["X_train_tonal", "X_test_tonal", "y_train_tonal", "y_test_tonal"],
            name="split_train_test_node",
        ),

        # 2. XGBoost
        node(
            func=train_xgb_optuna,
            inputs=["X_train_tonal", "y_train_tonal", "X_test_tonal", "y_test_tonal"],
            outputs="xgb_model_tonal",
            name="train_xgb_node",
        ),
    ])