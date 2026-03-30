"""
Pipeline d'inférence tonal.

Reçoit de nouvelles données before et prédit les after.

Étapes :
1. Validation des colonnes before
2. Conversion en numérique
3. Nettoyage des lignes invalides
4. Ajout des features (PTA + pente)
5. Prédiction avec le modèle entraîné
"""

from kedro.pipeline import Pipeline, node
 
from gain_project.nodes.tonal.inference import (
    validate_input,
    convert_numeric,
    clean_input,
    predict,
)
 
 
def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
 
        node(
            func=validate_input,
            inputs="inference_input_tonal",
            outputs="validated_input_tonal",
            name="validate_inference_input_node",
        ),
 
        node(
            func=convert_numeric,
            inputs="validated_input_tonal",
            outputs="numeric_input_tonal",
            name="convert_inference_numeric_node",
        ),
 
        node(
            func=clean_input,
            inputs="numeric_input_tonal",
            outputs=["clean_input_tonal", "invalid_rows_tonal"],
            name="clean_inference_input_node",
        ),
 
        node(
            func=predict,
            inputs=["xgb_model_tonal", "clean_input_tonal"],
            outputs="inference_predictions_tonal",
            name="predict_node",
        ),
    ])
 