"""
Pipeline de Data Engineering pour l'audiométrie vocale.

Même logique que le tonal :
1. Vérification des colonnes WRS
2. Conversion en numérique
3. Suppression des lignes incohérentes (< 0% ou > 100%)
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.vocal.preprocess import (
    validate_columns,
    convert_to_numeric,
    remove_invalid_rows,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=validate_columns,
            inputs="raw_vocal_data",
            outputs="validated_vocal_data",
            name="vocal_validate_columns_node",
        ),

        node(
            func=convert_to_numeric,
            inputs="validated_vocal_data",
            outputs="numeric_vocal_data",
            name="vocal_convert_numeric_node",
        ),

        node(
            func=remove_invalid_rows,
            inputs="numeric_vocal_data",
            outputs="clean_vocal_data",
            name="vocal_remove_invalid_rows_node",
        ),
    ])