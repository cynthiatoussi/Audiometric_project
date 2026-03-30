"""
Pipeline de Data Engineering pour l'audiométrie tonale.

Prépare les données brutes générées par le simulateur
avant l'étape de feature engineering et d'entraînement.

3 étapes successives :
1. Vérification des colonnes attendues
2. Conversion en format numérique
3. Suppression des lignes incohérentes
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.tonal.preprocess import (
    validate_columns,
    convert_to_numeric,
    remove_invalid_rows,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=validate_columns,
            inputs="raw_tonal_data",
            outputs="validated_tonal_data",
            name="validate_columns_node",
        ),

        node(
            func=convert_to_numeric,
            inputs="validated_tonal_data",
            outputs="numeric_tonal_data",
            name="convert_to_numeric_node",
        ),

        node(
            func=remove_invalid_rows,
            inputs="numeric_tonal_data",
            outputs="clean_tonal_data",
            name="remove_invalid_rows_node",
        ),
    ])