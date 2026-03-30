"""
Pipeline de feature engineering tonal.

Sépare les features (X) et la cible (y)
avec ajout du PTA et de la pente.
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.tonal.split import split_features_targets


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=split_features_targets,
            inputs="clean_tonal_data",
            outputs=["X_tonal", "y_tonal"],
            name="split_features_targets_node",
        ),
    ])