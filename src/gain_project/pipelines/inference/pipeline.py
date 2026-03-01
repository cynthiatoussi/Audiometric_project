from kedro.pipeline import Pipeline, node
from gain_project.nodes.inference import (
    validate_input,
    convert_numeric,
    detect_invalid_rows,
    predict,
)


def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline([

        node(
            func=validate_input,
            inputs="inference_input",
            outputs="validated_input",
        ),

        node(
            func=convert_numeric,
            inputs="validated_input",
            outputs="numeric_input",
        ),

        node(
            func=detect_invalid_rows,
            inputs="numeric_input",
            outputs="invalid_rows_inference",
        ),

        node(
            func=predict,
            inputs=["trained_model", "numeric_input"],
            outputs="inference_predictions",
        ),
    ])