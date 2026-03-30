"""
Pipeline d'inférence vocal.

Reçoit de nouvelles données WRS before (scores à 30-90 dB)
et prédit les WRS after.
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.vocal.inference import (
    validate_input,
    convert_numeric,
    clean_input,
    predict,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=validate_input,
            inputs="inference_input_vocal",
            outputs="validated_input_vocal",
            name="vocal_validate_inference_node",
        ),

        node(
            func=convert_numeric,
            inputs="validated_input_vocal",
            outputs="numeric_input_vocal",
            name="vocal_convert_inference_numeric_node",
        ),

        node(
            func=clean_input,
            inputs="numeric_input_vocal",
            outputs=["clean_input_vocal", "invalid_rows_vocal"],
            name="vocal_clean_inference_node",
        ),

        node(
            func=predict,
            inputs=["rf_vocal_model", "clean_input_vocal"],
            outputs="inference_predictions_vocal",
            name="vocal_predict_node",
        ),
    ])