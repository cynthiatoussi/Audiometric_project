"""
Pipeline de génération des données tonales.

Appelle le simulateur du professeur avec le nombre
d'échantillons défini dans parameters.yml.
"""

from kedro.pipeline import Pipeline, node

from gain_project.nodes.tonal.simulator import generate_tonal_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([

        node(
            func=generate_tonal_dataset,
            inputs="params:tonal.n_samples",
            outputs="raw_tonal_data",
            name="generate_tonal_data_node",
        ),
    ])