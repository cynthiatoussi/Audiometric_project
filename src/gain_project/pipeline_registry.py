"""
Pipeline registry.
"""

from kedro.pipeline import Pipeline
from gain_project.pipelines.data_engineering import create_pipeline as de_pipeline
from gain_project.pipelines.feature_engineering import create_pipeline as fe_pipeline
from gain_project.pipelines.training import create_pipeline as tr_pipeline
from gain_project.pipelines.inference import create_pipeline as inf_pipeline


def register_pipelines() -> dict[str, Pipeline]:
    return {
    "data_engineering": de_pipeline(),
    "feature_engineering": fe_pipeline(),
    "training": tr_pipeline(),
    "inference": inf_pipeline(),
    "__default__": de_pipeline() + fe_pipeline() + tr_pipeline(),
}