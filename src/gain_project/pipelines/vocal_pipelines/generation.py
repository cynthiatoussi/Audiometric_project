"""
Pipeline de génération des données vocales.

Appelle le simulateur vocal avec le nombre
d'échantillons défini dans parameters.yml.
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.vocal.simulator import generate_vocal_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=generate_vocal_dataset,
            inputs="params:vocal.n_samples",
            outputs="raw_vocal_data",
            name="vocal_generate_data_node",
        ),
    ])