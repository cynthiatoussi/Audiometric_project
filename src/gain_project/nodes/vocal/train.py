"""
Node d'entraînement pour l'audiométrie vocale.

Ce module :
- Sépare les données en train/test (80/20)
- Entraîne un Random Forest multi-output natif avec Optuna
- Log les résultats dans MLflow

Random Forest sklearn gère nativement le multi-output
(chaque arbre prédit les 7 WRS simultanément),
pas besoin de MultiOutputRegressor contrairement au XGBoost du tonal.
"""

import numpy as np
import optuna
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
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

    log.info("Split vocal : %d train, %d test", len(X_train), len(X_test))

    return X_train, X_test, y_train, y_test


def train_rf_optuna(X_train, y_train, X_test, y_test):
    """
    Entraîne un Random Forest multi-output avec Optuna.

    1. Recherche des meilleurs hyperparamètres (30 essais)
    2. Entraînement final avec les meilleurs paramètres
    3. Logging complet dans MLflow

    Retourne : le modèle entraîné
    """
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("vocal_wrs_prediction")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }

        model = RandomForestRegressor(random_state=42, n_jobs=-1, **params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return r2_score(y_test, preds)

    # Optimisation
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    log.info("RF Vocal — Meilleurs params : %s", best_params)
    log.info("RF Vocal — Meilleur R² Optuna : %.4f", study.best_value)

    # Entraînement final + logging MLflow
    with mlflow.start_run(run_name="RF_Optuna_Vocal"):

        model = RandomForestRegressor(
            random_state=42, n_jobs=-1, **best_params
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Params
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForest_MultiOutput")
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_train", X_train.shape[0])

        # Métriques
        mlflow.log_metric("r2", float(r2))
        mlflow.log_metric("rmse", float(rmse))
        mlflow.log_metric("mae", float(mae))

        # Modèle
        signature = infer_signature(X_test, preds)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name="Vocal_RF_Model",
        )

        log.info("RF Vocal — R²: %.4f, RMSE: %.2f, MAE: %.2f", r2, rmse, mae)

    return model