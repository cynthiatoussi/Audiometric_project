"""
Node d'entraînement pour l'audiométrie tonale.

Ce module :
- Sépare les données en train/test (80/20)
- Entraîne un XGBoost multi-output avec Optuna
- Log les résultats dans MLflow
"""

import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from mlflow.models.signature import infer_signature
import logging

log = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.WARNING)


def split_train_test(X, y):
    """
    Sépare les données en train/test (80/20).

    Retourne : X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    log.info("Split : %d train, %d test", len(X_train), len(X_test))

    return X_train, X_test, y_train, y_test


def train_xgb_optuna(X_train, y_train, X_test, y_test):
    """
    Entraîne un XGBoost multi-output avec Optuna.

    1. Recherche des meilleurs hyperparamètres (30 essais)
    2. Entraînement final avec les meilleurs paramètres
    3. Logging complet dans MLflow

    Retourne : le modèle entraîné
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("tonal_gain_prediction")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

        model = MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                **params,
            )
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return r2_score(y_test, preds)

    # Optimisation
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    log.info("XGBoost — Meilleurs params : %s", best_params)
    log.info("XGBoost — Meilleur R² Optuna : %.4f", study.best_value)

    # Entraînement final + logging MLflow
    with mlflow.start_run(run_name="XGBoost_Optuna"):

        model = MultiOutputRegressor(
            XGBRegressor(
                objective="reg:squarederror",
                tree_method="hist",
                random_state=42,
                **best_params,
            )
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "XGBoost_MultiOutput")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", X_train.shape[0])

        mlflow.log_metric("r2", float(r2))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))

        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="Tonal_XGBoost_Model",
        )

        log.info("XGBoost — R²: %.4f, RMSE: %.2f, MAE: %.2f", r2, rmse, mae)

    return model