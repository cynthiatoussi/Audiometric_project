"""
Pipeline de data engineering.

Ce pipeline :
1. Vérifie la présence des colonnes attendues
2. Convertit les colonnes en numérique
3. Détecte les lignes invalides
4. Sépare les données valides des données invalides
"""

from kedro.pipeline import Pipeline, node

# On importe les fonctions définies dans nodes.py
from gain_project.nodes.preprocess import (
    validate_columns,
    convert_to_numeric,
    detect_invalid_rows,
    split_valid_invalid,
)


def create_pipeline(**kwargs) -> Pipeline:
    """
    Crée le pipeline data_engineering.

    L'ordre des nodes est important :
    - On valide d'abord la structure
    - Puis on convertit en numérique
    - Ensuite on détecte les erreurs
    - Enfin on garde uniquement les lignes propres
    """

    return Pipeline([

        # Node 1 : Vérification des colonnes obligatoires
        node(
            func=validate_columns,            # Fonction appelée
            inputs="raw_tonal_data",          # Dataset d'entrée (catalog)
            outputs="validated_data",         # Dataset intermédiaire
            name="validate_columns_node",     # Nom du node
        ),

        # Node 2 : Conversion en numérique
        node(
            func=convert_to_numeric,
            inputs="validated_data",
            outputs="numeric_data",
            name="convert_to_numeric_node",
        ),

        # Node 3 : Détection des lignes invalides
        node(
            func=detect_invalid_rows,
            inputs="numeric_data",
            outputs="invalid_rows",           # Sauvegardé dans le catalog
            name="detect_invalid_rows_node",
        ),

        # Node 4 : Séparation des données valides
        node(
            func=split_valid_invalid,
            inputs="numeric_data",
            outputs="clean_tonal_data",       # Sauvegardé dans le catalog
            name="split_valid_invalid_node",
        ),
    ])
