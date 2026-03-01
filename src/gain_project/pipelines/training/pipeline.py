"""
Pipeline training.

Entraîne le modèle et calcule les métriques.
"""

from kedro.pipeline import Pipeline, node
from gain_project.nodes.train import train_model
from gain_project.nodes.evaluate import evaluate_model

def create_pipeline(**kwargs) -> Pipeline:

    return Pipeline([

        node(
            func=train_model,
            inputs=["X_train", "y_train"],
            outputs="trained_model",
            name="train_model_node",
        ),

        node(
            func=evaluate_model,
            inputs=["trained_model", "X_test", "y_test"],
            outputs="model_metrics",
            name="evaluate_model_node",
        ),
    ])