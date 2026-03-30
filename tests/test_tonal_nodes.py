"""
Tests unitaires pour les nodes tonal.
Exécution : pytest tests/
"""

import pytest
import pandas as pd
import numpy as np


# =============================================
# TESTS PREPROCESS TONAL
# =============================================

from gain_project.nodes.tonal.preprocess import (
    validate_columns,
    convert_to_numeric,
    remove_invalid_rows,
    REQUIRED_COLUMNS,
    MAX_DB,
)


class TestTonalValidateColumns:
    """Tests pour validate_columns."""

    def test_valid_dataframe(self):
        """Accepte un DataFrame avec toutes les colonnes requises."""
        df = pd.DataFrame(
            np.random.randint(0, 80, size=(5, 14)),
            columns=REQUIRED_COLUMNS,
        )
        result = validate_columns(df)
        assert list(result.columns) == REQUIRED_COLUMNS
        assert len(result) == 5

    def test_missing_column_raises(self):
        """Lève une erreur si une colonne manque."""
        df = pd.DataFrame({"before_exam_125_Hz": [10, 20]})
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            validate_columns(df)

    def test_extra_columns_ignored(self):
        """Les colonnes supplémentaires sont ignorées."""
        df = pd.DataFrame(
            np.random.randint(0, 80, size=(3, 14)),
            columns=REQUIRED_COLUMNS,
        )
        df["extra_column"] = 999
        result = validate_columns(df)
        assert "extra_column" not in result.columns


class TestTonalConvertToNumeric:
    """Tests pour convert_to_numeric."""

    def test_numeric_values_unchanged(self):
        """Les valeurs numériques restent identiques."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.5, 5.5, 6.5]})
        result = convert_to_numeric(df)
        assert result["a"].tolist() == [1, 2, 3]

    def test_letters_become_nan(self):
        """Les lettres sont converties en NaN."""
        df = pd.DataFrame({"a": ["10", "ABC", "30"]})
        result = convert_to_numeric(df)
        assert pd.isna(result["a"].iloc[1])
        assert result["a"].iloc[0] == 10

    def test_mixed_types(self):
        """Gère un mix de types (int, float, str, NaN)."""
        df = pd.DataFrame({"a": [10, 20.5, "X", np.nan]})
        result = convert_to_numeric(df)
        assert result["a"].iloc[0] == 10
        assert result["a"].iloc[1] == 20.5
        assert pd.isna(result["a"].iloc[2])
        assert pd.isna(result["a"].iloc[3])


class TestTonalRemoveInvalidRows:
    """Tests pour remove_invalid_rows."""

    def test_clean_data_unchanged(self):
        """Les données propres passent intactes."""
        df = pd.DataFrame({"a": [10, 50, 80], "b": [20, 60, 90]})
        result = remove_invalid_rows(df)
        assert len(result) == 3

    def test_nan_rows_removed(self):
        """Les lignes avec NaN sont supprimées."""
        df = pd.DataFrame({"a": [10, np.nan, 30], "b": [20, 40, 60]})
        result = remove_invalid_rows(df)
        assert len(result) == 2

    def test_outlier_rows_removed(self):
        """Les lignes avec valeurs > MAX_DB sont supprimées."""
        df = pd.DataFrame({"a": [10, 150, 30], "b": [20, 40, 60]})
        result = remove_invalid_rows(df)
        assert len(result) == 2

    def test_empty_after_cleaning(self):
        """Retourne un DataFrame vide si tout est invalide."""
        df = pd.DataFrame({"a": [np.nan, 150], "b": [np.nan, 200]})
        result = remove_invalid_rows(df)
        assert len(result) == 0

    def test_index_reset(self):
        """L'index est réinitialisé après nettoyage."""
        df = pd.DataFrame({"a": [np.nan, 50, 80], "b": [20, 60, 90]})
        result = remove_invalid_rows(df)
        assert result.index.tolist() == [0, 1]


# =============================================
# TESTS SPLIT TONAL
# =============================================

from gain_project.nodes.tonal.split import (
    split_features_targets,
    BEFORE_COLUMNS,
    AFTER_COLUMNS,
)


class TestTonalSplit:
    """Tests pour split_features_targets."""

    def _make_df(self, n=10):
        """Crée un DataFrame tonal de test."""
        data = np.random.randint(0, 100, size=(n, 14))
        return pd.DataFrame(data, columns=BEFORE_COLUMNS + AFTER_COLUMNS)

    def test_split_shapes(self):
        """X et y ont les bonnes dimensions."""
        df = self._make_df(20)
        X, y = split_features_targets(df)
        assert X.shape == (20, 7)
        assert y.shape == (20, 7)

    def test_split_columns(self):
        """X contient les before, y contient les after."""
        df = self._make_df()
        X, y = split_features_targets(df)
        assert list(X.columns) == BEFORE_COLUMNS
        assert list(y.columns) == AFTER_COLUMNS

    def test_split_no_data_leak(self):
        """Aucune colonne after dans X."""
        df = self._make_df()
        X, y = split_features_targets(df)
        for col in AFTER_COLUMNS:
            assert col not in X.columns