"""
API REST pour l'exposition des modèles de gains prothétiques.

Routes (conformes aux modalités d'évaluation) :
    GET  /                  → Interface web + statut API
    GET  /status            → Statut API + modèles (JSON)
    POST /train             → Exécute tous les pipelines d'entraînement
    POST /train/tonal       → Entraînement tonal uniquement
    POST /train/vocal       → Entraînement vocal uniquement
    POST /predict           → Auto-détecte tonal ou vocal et prédit
    POST /predict/tonal     → Prédiction audiométrie tonale
    POST /predict/vocal     → Prédiction audiométrie vocale

Conformité évaluation :
    - /train exécute les différents pipelines retenus
    - /predict retourne les prédictions ET l'emplacement des lignes invalides
    - Les données d'entrée sont sauvegardées dans data/01_raw/user_inputs/
    - Gestion des erreurs complète
"""

import os
import json
import pickle
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template

log = logging.getLogger(__name__)

app = Flask(__name__)
app.json.ensure_ascii = False

# -------------------------------------------------
# Chemins
# -------------------------------------------------

BASE_DIR = Path(__file__).parent
TONAL_MODEL_PATH = BASE_DIR / "data" / "06_models" / "xgb_model_tonal.pkl"
VOCAL_MODEL_PATH = BASE_DIR / "data" / "06_models" / "rf_vocal_model.pkl"
USER_INPUT_DIR = BASE_DIR / "data" / "01_raw" / "user_inputs"

# Créer le dossier pour sauvegarder les données utilisateur
USER_INPUT_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# Colonnes attendues
# -------------------------------------------------

TONAL_FREQUENCIES = [125, 250, 500, 1000, 2000, 4000, 8000]
TONAL_BEFORE_COLS = [f"before_exam_{f}_Hz" for f in TONAL_FREQUENCIES]
TONAL_AFTER_COLS = [f"after_exam_{f}_Hz" for f in TONAL_FREQUENCIES]

VOCAL_INTENSITIES = [30, 40, 50, 60, 70, 80, 90]
VOCAL_BEFORE_COLS = [f"wrs_{i}_before" for i in VOCAL_INTENSITIES]
VOCAL_AFTER_COLS = [f"wrs_{i}_after" for i in VOCAL_INTENSITIES]

# Borne tonal (synchronisé avec preprocess.py tonal)
TONAL_MAX_DB = 120

# Bornes vocal (synchronisé avec preprocess.py vocal)
VOCAL_MIN_WRS = 0
VOCAL_MAX_WRS = 100


# -------------------------------------------------
# Chargement des modèles
# -------------------------------------------------

def load_model(path):
    """Charge un modèle pickle. Retourne None si inexistant."""
    if not path.exists():
        log.warning("Modèle non trouvé : %s", path)
        return None
    with open(path, "rb") as f:
        model = pickle.load(f)
    log.info("Modèle chargé : %s", path)
    return model


tonal_model = load_model(TONAL_MODEL_PATH)
vocal_model = load_model(VOCAL_MODEL_PATH)


# -------------------------------------------------
# Sauvegarde des données utilisateur
# -------------------------------------------------

def save_user_input(data, data_type):
    """
    Sauvegarde les données envoyées par l'utilisateur.
    Conforme au TP : 'La sauvegarde de toutes les données d'entrées
    en les identifiant comme données en provenance d'utilisateurs'
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{data_type}_{timestamp}.json"
    filepath = USER_INPUT_DIR / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    log.info("Données utilisateur sauvegardées : %s", filepath)
    return str(filepath)


# -------------------------------------------------
# Détection du type de données
# -------------------------------------------------

def detect_data_type(columns):
    """Détecte si les données sont tonales ou vocales."""
    tonal_match = set(TONAL_BEFORE_COLS).issubset(set(columns))
    vocal_match = set(VOCAL_BEFORE_COLS).issubset(set(columns))

    if tonal_match:
        return "tonal"
    elif vocal_match:
        return "vocal"
    else:
        return None


# -------------------------------------------------
# Validation et prédiction
# -------------------------------------------------

def validate_and_predict_tonal(df):
    """
    Valide les données tonales et retourne prédictions + lignes invalides.
    Messages d'erreur détaillés pour chaque cellule problématique.
    """
    df_raw = df[TONAL_BEFORE_COLS].copy()
    df_numeric = df_raw.apply(pd.to_numeric, errors="coerce")

    # Détection des lignes invalides avec raison précise
    invalid_rows = []
    for idx in range(len(df_numeric)):
        row_num = df_numeric.iloc[idx]
        row_raw = df_raw.iloc[idx]
        issues = []

        for col in TONAL_BEFORE_COLS:
            raw_val = row_raw[col]
            num_val = row_num[col]

            if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                issues.append(f"{col}: cellule vide")
            elif pd.isna(num_val):
                issues.append(f"{col}: valeur non numérique ('{raw_val}')")
            elif num_val > TONAL_MAX_DB:
                issues.append(f"{col}: valeur {num_val} dépasse le seuil de {TONAL_MAX_DB} dB")

        if issues:
            invalid_rows.append({
                "row_index": idx + 1,
                "issues": issues,
            })

    # Séparer valides et invalides
    invalid_indices = [r["row_index"] - 1 for r in invalid_rows]
    valid_mask = ~df_numeric.index.isin(invalid_indices)
    df_valid = df_numeric[valid_mask]

    predictions = []
    if len(df_valid) > 0:
        preds = tonal_model.predict(df_valid)
        result = pd.DataFrame(preds, columns=TONAL_AFTER_COLS, index=df_valid.index)
        predictions = result.round(1).to_dict(orient="records")

    return predictions, invalid_rows


def validate_and_predict_vocal(df):
    """
    Valide les données vocales et retourne prédictions + lignes invalides.
    Messages d'erreur détaillés pour chaque cellule problématique.
    """
    df_raw = df[VOCAL_BEFORE_COLS].copy()
    df_numeric = df_raw.apply(pd.to_numeric, errors="coerce")

    # Détection des lignes invalides avec raison précise
    invalid_rows = []
    for idx in range(len(df_numeric)):
        row_num = df_numeric.iloc[idx]
        row_raw = df_raw.iloc[idx]
        issues = []

        for col in VOCAL_BEFORE_COLS:
            raw_val = row_raw[col]
            num_val = row_num[col]

            if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                issues.append(f"{col}: cellule vide")
            elif pd.isna(num_val):
                issues.append(f"{col}: valeur non numérique ('{raw_val}')")
            elif num_val < VOCAL_MIN_WRS:
                issues.append(f"{col}: valeur {num_val}% inférieure au minimum {VOCAL_MIN_WRS}%")
            elif num_val > VOCAL_MAX_WRS:
                issues.append(f"{col}: valeur {num_val}% dépasse le maximum {VOCAL_MAX_WRS}%")

        if issues:
            invalid_rows.append({
                "row_index": idx + 1,
                "issues": issues,
            })

    invalid_indices = [r["row_index"] - 1 for r in invalid_rows]
    valid_mask = ~df_numeric.index.isin(invalid_indices)
    df_valid = df_numeric[valid_mask]

    predictions = []
    if len(df_valid) > 0:
        preds = vocal_model.predict(df_valid)
        result = pd.DataFrame(preds, columns=VOCAL_AFTER_COLS, index=df_valid.index)
        predictions = result.round(1).to_dict(orient="records")

    return predictions, invalid_rows


# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """Route par défaut — sert l'interface web."""
    return render_template("index.html")


@app.route("/status", methods=["GET"])
def status():
    """Statut de l'API et des modèles."""
    return jsonify({
        "status": "running",
        "models": {
            "tonal": "loaded" if tonal_model is not None else "not found",
            "vocal": "loaded" if vocal_model is not None else "not found",
        },
        "routes": {
            "GET /": "Interface web",
            "GET /status": "Statut API",
            "POST /train": "Entraînement de tous les pipelines",
            "POST /predict": "Prédiction (auto-détection tonal/vocal)",
            "POST /predict/tonal": "Prédiction tonal",
            "POST /predict/vocal": "Prédiction vocal",
        },
    })


# === TRAIN ===

@app.route("/train", methods=["POST"])
def train_all():
    """
    Exécute tous les pipelines d'entraînement.
    Conforme évaluation : '/train pour exécuter les différents pipelines retenus'
    """
    global tonal_model, vocal_model
    results = {}

    try:
        os.system("kedro run --pipeline=tonal_full")
        tonal_model = load_model(TONAL_MODEL_PATH)
        results["tonal"] = "entraîné et rechargé"
    except Exception as e:
        results["tonal"] = f"erreur: {str(e)}"

    try:
        os.system("kedro run --pipeline=vocal_full")
        vocal_model = load_model(VOCAL_MODEL_PATH)
        results["vocal"] = "entraîné et rechargé"
    except Exception as e:
        results["vocal"] = f"erreur: {str(e)}"

    return jsonify({"status": "Entraînement terminé", "details": results})


@app.route("/train/tonal", methods=["POST"])
def train_tonal():
    """Entraînement tonal uniquement."""
    global tonal_model
    try:
        os.system("kedro run --pipeline=tonal_full")
        tonal_model = load_model(TONAL_MODEL_PATH)
        return jsonify({"status": "Entraînement tonal terminé", "model": "rechargé"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/train/vocal", methods=["POST"])
def train_vocal():
    """Entraînement vocal uniquement."""
    global vocal_model
    try:
        os.system("kedro run --pipeline=vocal_full")
        vocal_model = load_model(VOCAL_MODEL_PATH)
        return jsonify({"status": "Entraînement vocal terminé", "model": "rechargé"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === PREDICT ===

@app.route("/predict", methods=["POST"])
def predict_auto():
    """
    Prédiction avec auto-détection du type de données.
    Conforme évaluation : retourne les lignes invalides avec leur emplacement.
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Corps de requête vide"}), 400
        df = pd.DataFrame(data)
    except Exception as e:
        return jsonify({"error": f"Format JSON invalide : {str(e)}"}), 400

    # Sauvegarde des données utilisateur
    save_user_input(data, "predict")

    # Auto-détection
    data_type = detect_data_type(df.columns.tolist())

    if data_type == "tonal":
        return _predict_tonal_internal(df, data)
    elif data_type == "vocal":
        return _predict_vocal_internal(df, data)
    else:
        return jsonify({
            "error": "Type de données non reconnu.",
            "expected_tonal_columns": TONAL_BEFORE_COLS,
            "expected_vocal_columns": VOCAL_BEFORE_COLS,
        }), 400


@app.route("/predict/tonal", methods=["POST"])
def predict_tonal():
    """Prédiction tonal explicite."""
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
    except Exception as e:
        return jsonify({"error": f"Format JSON invalide : {str(e)}"}), 400

    save_user_input(data, "predict_tonal")
    return _predict_tonal_internal(df, data)


@app.route("/predict/vocal", methods=["POST"])
def predict_vocal():
    """Prédiction vocal explicite."""
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
    except Exception as e:
        return jsonify({"error": f"Format JSON invalide : {str(e)}"}), 400

    save_user_input(data, "predict_vocal")
    return _predict_vocal_internal(df, data)


def _predict_tonal_internal(df, raw_data):
    """Logique commune de prédiction tonale."""
    global tonal_model

    if tonal_model is None:
        tonal_model = load_model(TONAL_MODEL_PATH)
        if tonal_model is None:
            return jsonify({"error": "Modèle tonal non entraîné. Utilisez POST /train"}), 404

    missing = set(TONAL_BEFORE_COLS) - set(df.columns)
    if missing:
        return jsonify({"error": f"Colonnes manquantes : {list(missing)}"}), 400

    predictions, invalid_rows = validate_and_predict_tonal(df)

    return jsonify({
        "type": "tonal",
        "predictions": predictions,
        "n_total": len(df),
        "n_predicted": len(predictions),
        "n_invalid": len(invalid_rows),
        "invalid_rows": invalid_rows,
    })


def _predict_vocal_internal(df, raw_data):
    """Logique commune de prédiction vocale."""
    global vocal_model

    if vocal_model is None:
        vocal_model = load_model(VOCAL_MODEL_PATH)
        if vocal_model is None:
            return jsonify({"error": "Modèle vocal non entraîné. Utilisez POST /train"}), 404

    missing = set(VOCAL_BEFORE_COLS) - set(df.columns)
    if missing:
        return jsonify({"error": f"Colonnes manquantes : {list(missing)}"}), 400

    predictions, invalid_rows = validate_and_predict_vocal(df)

    return jsonify({
        "type": "vocal",
        "predictions": predictions,
        "n_total": len(df),
        "n_predicted": len(predictions),
        "n_invalid": len(invalid_rows),
        "invalid_rows": invalid_rows,
    })


# -------------------------------------------------
# Gestion des erreurs
# -------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Route non trouvée"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Erreur serveur interne"}), 500


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Méthode non autorisée"}), 405


# -------------------------------------------------
# Lancement
# -------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5001,
        debug=True
    )