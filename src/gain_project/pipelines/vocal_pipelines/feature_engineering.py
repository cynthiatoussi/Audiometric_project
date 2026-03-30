"""
Pipeline de feature engineering vocal.

Sépare les features (WRS before) et la cible (WRS after).
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.vocal.split import split_features_targets


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=split_features_targets,
            inputs="clean_vocal_data",
            outputs=["X_vocal", "y_vocal"],
            name="vocal_split_features_targets_node",
        ),
    ])