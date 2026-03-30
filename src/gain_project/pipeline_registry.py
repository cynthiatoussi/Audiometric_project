"""
Pipeline registry du projet.

TONAL (XGBoost + Optuna) :
- tonal_generation   -> Génération audiogrammes tonaux (7 fréquences)
- tonal_de           -> Data engineering (validation, nettoyage)
- tonal_fe           -> Feature engineering (PTA, pente, split X/y)
- tonal_training     -> Split train/test + XGBoost + Optuna + MLflow
- tonal_inference    -> Prédiction sur nouvelles données
- tonal_full         -> Pipeline complet (generation -> training)

VOCAL (Random Forest + Optuna) :
- vocal_generation   -> Génération courbes WRS sigmoïdes (5 profils)
- vocal_de           -> Data engineering (validation, nettoyage 0-100%)
- vocal_fe           -> Feature engineering (split X/y)
- vocal_training     -> Split train/test + RF + Optuna + MLflow
- vocal_inference    -> Prédiction sur nouvelles données
- vocal_full         -> Pipeline complet (generation -> training)
"""

# ==============================
# TONAL PIPELINES
# ==============================

from gain_project.pipelines.tonal_pipelines.generation import (
    create_pipeline as tonal_generation,
)
from gain_project.pipelines.tonal_pipelines.data_engineering import (
    create_pipeline as tonal_de,
)
from gain_project.pipelines.tonal_pipelines.feature_engineering import (
    create_pipeline as tonal_fe,
)
from gain_project.pipelines.tonal_pipelines.training import (
    create_pipeline as tonal_training,
)
from gain_project.pipelines.tonal_pipelines.inference import (
    create_pipeline as tonal_inference,
)

# ==============================
# VOCAL PIPELINES
# ==============================

from gain_project.pipelines.vocal_pipelines.generation import (
    create_pipeline as vocal_generation,
)
from gain_project.pipelines.vocal_pipelines.data_engineering import (
    create_pipeline as vocal_de,
)
from gain_project.pipelines.vocal_pipelines.feature_engineering import (
    create_pipeline as vocal_fe,
)
from gain_project.pipelines.vocal_pipelines.training import (
    create_pipeline as vocal_training,
)
from gain_project.pipelines.vocal_pipelines.inference import (
    create_pipeline as vocal_inference,
)


# ==============================
# REGISTRY
# ==============================

def register_pipelines():
    return {

        # -------- TONAL --------
        "tonal_generation": tonal_generation(),
        "tonal_de": tonal_de(),
        "tonal_fe": tonal_fe(),
        "tonal_training": tonal_training(),
        "tonal_inference": tonal_inference(),
        "tonal_full": (
            tonal_generation()
            + tonal_de()
            + tonal_fe()
            + tonal_training()
        ),

        # -------- VOCAL --------
        "vocal_generation": vocal_generation(),
        "vocal_de": vocal_de(),
        "vocal_fe": vocal_fe(),
        "vocal_training": vocal_training(),
        "vocal_inference": vocal_inference(),
        "vocal_full": (
            vocal_generation()
            + vocal_de()
            + vocal_fe()
            + vocal_training()
        ),

        # -------- FULL PROJECT --------
        "full": (
            tonal_generation()
            + tonal_de()
            + tonal_fe()
            + tonal_training()
            + vocal_generation()
            + vocal_de()
            + vocal_fe()
            + vocal_training()
        ),

        # -------- DEFAULT --------
        "__default__": (
            tonal_generation()
            + tonal_de()
            + tonal_fe()
            + tonal_training()
            + vocal_generation()
            + vocal_de()
            + vocal_fe()
            + vocal_training()
        ),
    }