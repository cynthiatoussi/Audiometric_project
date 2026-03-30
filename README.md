# 🔊 PrediGain — Prédiction de Gains Prothétiques en Audiométrie

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/API-Flask-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com)
[![Tests](https://img.shields.io/badge/tests-38%20passed-brightgreen.svg)]()

---

## 📋 Description

PrediGain est un projet d'**industrialisation de modèles ML** pour la prédiction de gains prothétiques en audiométrie. Il prédit l'amélioration auditive d'un patient après la pose d'une prothèse auditive à partir de ses mesures **avant** prothèse.

Deux types d'audiométrie sont traités :

- **Audiométrie tonale** : prédiction des seuils auditifs (dB HL) à 7 fréquences (125 Hz → 8000 Hz)
- **Audiométrie vocale** : prédiction des scores d'intelligibilité (% WRS) à 7 intensités (30 → 90 dB)

---

## 🏗️ Architecture du projet

```
gain-project/
│
├── app.py                              ← API REST Flask
├── Dockerfile                          ← Conteneurisation Docker
├── .dockerignore                       ← Fichiers exclus du Docker
├── requirements.txt                    ← Dépendances Python
├── pyproject.toml                      ← Configuration Kedro + pytest
│
├── templates/
│   └── index.html                      ← Interface web PrediGain
│
├── tests/
│   ├── __init__.py
│   ├── test_tonal_nodes.py             ← 13 tests nodes tonal
│   ├── test_vocal_nodes.py             ← 9 tests nodes vocal
│   └── test_api.py                     ← 16 tests API Flask
│
├── .github/workflows/
│   └── ci.yml                          ← CI/CD GitHub Actions
│
├── conf/base/
│   ├── catalog.yml                     ← Catalogue de données Kedro
│   └── parameters.yml                  ← Paramètres (n_samples: 50000)
│
├── data/
│   ├── 01_raw/                         ← Données brutes générées
│   │   └── user_inputs/                ← Données reçues via l'API
│   ├── 02_intermediate/                ← Données nettoyées
│   ├── 03_primary/                     ← Features et targets (X, y)
│   ├── 05_model_input/                 ← Train/test splits
│   ├── 06_models/                      ← Modèles entraînés
│   │   ├── xgb_model_tonal.pkl         ← XGBoost (R²=0.943)
│   │   └── rf_vocal_model.pkl          ← Random Forest (R²=0.862)
│   ├── 07_model_output/                ← Prédictions
│   └── 08_reporting/                   ← Métriques et rapports
│
└── src/gain_project/
    ├── pipeline_registry.py            ← Registre des 12 pipelines
    ├── nodes/
    │   ├── tonal/                      ← simulator, preprocess, split, train, inference
    │   └── vocal/                      ← simulator, preprocess, split, train, inference
    └── pipelines/
        ├── tonal_pipelines/            ← generation, data_engineering, feature_engineering, training, inference
        └── vocal_pipelines/            ← idem
```

---

## 🤖 Modèles de Machine Learning

| | Tonal | Vocal |
|---|---|---|
| **Modèle** | XGBoost + MultiOutputRegressor | Random Forest multi-output natif |
| **Optimisation** | Optuna (30 trials) | Optuna (30 trials) |
| **Features** | 7 fréquences brutes (before) | 7 scores WRS bruts (before) |
| **Cibles** | 7 seuils after | 7 WRS after |
| **R²** | 0.943 | 0.862 |
| **MAE** | 3.55 dB | 4.21 % |
| **RMSE** | 7.50 dB | 6.55 % |
| **Tracking** | MLflow + Model Registry | MLflow + Model Registry |

**Pourquoi deux modèles différents ?**

- **XGBoost pour le tonal** : les 7 fréquences sont indépendantes. Le boosting séquentiel corrige les erreurs itérativement. Chaque fréquence a son propre sous-modèle via MultiOutputRegressor.
- **Random Forest pour le vocal** : les 7 scores WRS suivent une courbe sigmoïde corrélée. RF multi-output natif capture les corrélations entre les 7 sorties dans chaque arbre.

---

## 🚀 Installation et utilisation

### Prérequis

- Python 3.11+
- Docker (pour le déploiement)

### Installation locale

```bash
git clone <url-du-repo>
cd gain-project
pip install -r requirements.txt
pip install -e .
```

### Entraînement des modèles

```bash
# Tout entraîner (tonal + vocal)
kedro run

# Tonal uniquement
kedro run --pipeline=tonal_full

# Vocal uniquement
kedro run --pipeline=vocal_full
```

### Lancer l'API

```bash
python app.py
# → http://localhost:5001
```

### Lancer les tests

```bash
pytest tests/ -v
# → 38 passed
```

### Visualisation des pipelines

```bash
kedro viz run --port 4000
# → http://localhost:4000
```

### Suivi MLflow

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

---

## 🐳 Docker

```bash
# Construire l'image
docker build -t gain-project .

# Lancer le container
docker run -p 5001:5001 gain-project

# → http://localhost:5001
```

---

## 📡 API REST

### Routes disponibles

| Méthode | Route | Description |
|---|---|---|
| GET | `/` | Interface web PrediGain |
| GET | `/status` | Statut de l'API et des modèles |
| POST | `/train` | Entraîne tous les pipelines |
| POST | `/train/tonal` | Entraîne le pipeline tonal |
| POST | `/train/vocal` | Entraîne le pipeline vocal |
| POST | `/predict` | Prédiction (auto-détecte tonal/vocal) |
| POST | `/predict/tonal` | Prédiction tonal explicite |
| POST | `/predict/vocal` | Prédiction vocal explicite |

### Exemples

**Prédiction tonal :**

```bash
curl -X POST http://localhost:5001/predict/tonal \
  -H "Content-Type: application/json" \
  -d '[{"before_exam_125_Hz":45,"before_exam_250_Hz":50,"before_exam_500_Hz":55,"before_exam_1000_Hz":60,"before_exam_2000_Hz":65,"before_exam_4000_Hz":70,"before_exam_8000_Hz":75}]'
```

**Prédiction vocal :**

```bash
curl -X POST http://localhost:5001/predict/vocal \
  -H "Content-Type: application/json" \
  -d '[{"wrs_30_before":5,"wrs_40_before":20,"wrs_50_before":55,"wrs_60_before":80,"wrs_70_before":90,"wrs_80_before":95,"wrs_90_before":92}]'
```

**Prédiction avec lignes invalides :**

```bash
curl -X POST http://localhost:5001/predict/tonal \
  -H "Content-Type: application/json" \
  -d '[{"before_exam_125_Hz":45,"before_exam_250_Hz":50,"before_exam_500_Hz":55,"before_exam_1000_Hz":60,"before_exam_2000_Hz":65,"before_exam_4000_Hz":70,"before_exam_8000_Hz":75},{"before_exam_125_Hz":null,"before_exam_250_Hz":50,"before_exam_500_Hz":null,"before_exam_1000_Hz":60,"before_exam_2000_Hz":65,"before_exam_4000_Hz":70,"before_exam_8000_Hz":75}]'
```

**Réponse :**

```json
{
  "type": "tonal",
  "n_total": 2,
  "n_predicted": 1,
  "n_invalid": 1,
  "predictions": [{"after_exam_125_Hz": 30.2, "...": "..."}],
  "invalid_rows": [
    {
      "row_index": 2,
      "issues": [
        "before_exam_125_Hz: cellule vide",
        "before_exam_500_Hz: cellule vide"
      ]
    }
  ]
}
```

Les lignes invalides sont identifiées par leur `row_index` (commence à 1) avec la cause précise de chaque cellule problématique :
- `cellule vide` : valeur null ou absente
- `valeur non numérique ('ABC')` : contient du texte
- `valeur 150 dépasse le seuil de 120 dB` : hors bornes

---

## 🧪 Tests

38 tests répartis en 3 fichiers :

| Fichier | Tests | Couverture |
|---|---|---|
| `test_tonal_nodes.py` | 13 | validate_columns, convert_to_numeric, remove_invalid_rows, split_features_targets |
| `test_vocal_nodes.py` | 9 | validate_columns, convert_to_numeric, remove_invalid_rows, split_features_targets |
| `test_api.py` | 16 | Routes /, /status, /predict/tonal, /predict/vocal, /predict auto, erreurs 404/405 |

---

## 🔄 CI/CD

GitHub Actions exécute automatiquement à chaque push :
1. Installation Python 3.11 + dépendances
2. `pytest tests/ -v` — si un test échoue, le pipeline est rouge
3. `docker build` — vérifie que l'image compile

---

## 🔧 Traitements appliqués aux données

1. **Validation** : vérification des colonnes requises
2. **Conversion** : `pd.to_numeric(errors='coerce')` — lettres → NaN
3. **Nettoyage** : suppression si NaN, > 120 dB (tonal) ou < 0% / > 100% (vocal)

---

## 📊 Ports utilisés

| Service | Port | Commande |
|---|---|---|
| Kedro Viz | 4000 | `kedro viz run --port 4000` |
| MLflow | 5000 | `mlflow ui --port 5000` |
| API Flask | 5001 | `python app.py` |

---