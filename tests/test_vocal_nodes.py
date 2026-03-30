"""
Tests unitaires pour les nodes vocal.
"""

import pytest
import pandas as pd
import numpy as np

from gain_project.nodes.vocal.preprocess import (
    validate_columns,
    convert_to_numeric,
    remove_invalid_rows,
    REQUIRED_COLUMNS,
    MIN_WRS,
    MAX_WRS,
)


class TestVocalValidateColumns:

    def test_valid_dataframe(self):
        df = pd.DataFrame(
            np.random.randint(0, 100, size=(5, 14)),
            columns=REQUIRED_COLUMNS,
        )
        result = validate_columns(df)
        assert len(result) == 5

    def test_missing_column_raises(self):
        df = pd.DataFrame({"wrs_30_before": [10, 20]})
        with pytest.raises(ValueError, match="Colonnes manquantes"):
            validate_columns(df)


class TestVocalConvertToNumeric:

    def test_letters_become_nan(self):
        df = pd.DataFrame({"a": ["50", "ABC", "80"]})
        result = convert_to_numeric(df)
        assert pd.isna(result["a"].iloc[1])


class TestVocalRemoveInvalidRows:

    def test_negative_removed(self):
        """Les scores négatifs sont supprimés."""
        df = pd.DataFrame({"a": [-5, 50, 80], "b": [20, 40, 60]})
        result = remove_invalid_rows(df)
        assert len(result) == 2

    def test_above_100_removed(self):
        """Les scores > 100% sont supprimés."""
        df = pd.DataFrame({"a": [50, 110, 80], "b": [20, 40, 60]})
        result = remove_invalid_rows(df)
        assert len(result) == 2

    def test_clean_data_passes(self):
        df = pd.DataFrame({"a": [0, 50, 100], "b": [20, 60, 80]})
        result = remove_invalid_rows(df)
        assert len(result) == 3


# =============================================
# TESTS SPLIT VOCAL
# =============================================

from gain_project.nodes.vocal.split import (
    split_features_targets,
    BEFORE_COLUMNS,
    AFTER_COLUMNS,
)


class TestVocalSplit:

    def _make_df(self, n=10):
        data = np.random.randint(0, 100, size=(n, 14))
        return pd.DataFrame(data, columns=BEFORE_COLUMNS + AFTER_COLUMNS)

    def test_split_shapes(self):
        df = self._make_df(15)
        X, y = split_features_targets(df)
        assert X.shape == (15, 7)
        assert y.shape == (15, 7)

    def test_split_columns(self):
        df = self._make_df()
        X, y = split_features_targets(df)
        assert list(X.columns) == BEFORE_COLUMNS
        assert list(y.columns) == AFTER_COLUMNS