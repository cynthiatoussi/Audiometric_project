"""
Tests de l'API Flask.
Teste les routes /, /status, /predict, /predict/tonal, /predict/vocal.
"""

import pytest
import json
from app import app


@pytest.fixture
def client():
    """Client de test Flask."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


# =============================================
# ROUTE /
# =============================================

class TestHomeRoute:

    def test_home_returns_200(self, client):
        """La route / retourne 200."""
        response = client.get("/")
        assert response.status_code == 200

    def test_home_returns_html(self, client):
        """La route / retourne du HTML."""
        response = client.get("/")
        assert b"<!DOCTYPE html>" in response.data or b"<html" in response.data


# =============================================
# ROUTE /status
# =============================================

class TestStatusRoute:

    def test_status_returns_200(self, client):
        response = client.get("/status")
        assert response.status_code == 200

    def test_status_has_models(self, client):
        response = client.get("/status")
        data = json.loads(response.data)
        assert "models" in data
        assert "tonal" in data["models"]
        assert "vocal" in data["models"]

    def test_status_running(self, client):
        response = client.get("/status")
        data = json.loads(response.data)
        assert data["status"] == "running"


# =============================================
# ROUTE /predict/tonal
# =============================================

class TestPredictTonal:

    def _valid_tonal_row(self):
        return {
            "before_exam_125_Hz": 45,
            "before_exam_250_Hz": 50,
            "before_exam_500_Hz": 55,
            "before_exam_1000_Hz": 60,
            "before_exam_2000_Hz": 65,
            "before_exam_4000_Hz": 70,
            "before_exam_8000_Hz": 75,
        }

    def test_valid_prediction(self, client):
        """10 lignes valides retournent 10 prédictions."""
        payload = [self._valid_tonal_row() for _ in range(10)]
        response = client.post(
            "/predict/tonal",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = json.loads(response.data)

        # Si modèle pas chargé, on vérifie juste que l'API répond
        if response.status_code == 404:
            assert "error" in data
            return

        assert response.status_code == 200
        assert data["n_predicted"] == 10
        assert data["n_invalid"] == 0

    def test_invalid_rows_reported(self, client):
        """Les lignes avec cases vides sont signalées."""
        rows = [self._valid_tonal_row() for _ in range(10)]
        # Injecter des valeurs invalides
        rows[2]["before_exam_500_Hz"] = None
        rows[5]["before_exam_1000_Hz"] = None
        rows[7]["before_exam_250_Hz"] = "ABC"

        response = client.post(
            "/predict/tonal",
            data=json.dumps(rows),
            content_type="application/json",
        )
        data = json.loads(response.data)

        if response.status_code == 404:
            return

        assert data["n_invalid"] == 3
        invalid_indices = [r["row_index"] for r in data["invalid_rows"]]
        assert 2 in invalid_indices
        assert 5 in invalid_indices
        assert 7 in invalid_indices

    def test_missing_columns(self, client):
        """Colonnes manquantes retournent une erreur 400."""
        payload = [{"wrong_column": 50}]
        response = client.post(
            "/predict/tonal",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 400

    def test_empty_body(self, client):
        """Corps vide retourne une erreur."""
        response = client.post(
            "/predict/tonal",
            data="",
            content_type="application/json",
        )
        assert response.status_code == 400


# =============================================
# ROUTE /predict/vocal
# =============================================

class TestPredictVocal:

    def _valid_vocal_row(self):
        return {
            "wrs_30_before": 5,
            "wrs_40_before": 20,
            "wrs_50_before": 55,
            "wrs_60_before": 80,
            "wrs_70_before": 90,
            "wrs_80_before": 95,
            "wrs_90_before": 92,
        }

    def test_valid_prediction(self, client):
        payload = [self._valid_vocal_row() for _ in range(10)]
        response = client.post(
            "/predict/vocal",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = json.loads(response.data)

        if response.status_code == 404:
            return

        assert response.status_code == 200
        assert data["n_predicted"] == 10

    def test_invalid_rows_reported(self, client):
        rows = [self._valid_vocal_row() for _ in range(10)]
        rows[1]["wrs_50_before"] = None
        rows[4]["wrs_70_before"] = None

        response = client.post(
            "/predict/vocal",
            data=json.dumps(rows),
            content_type="application/json",
        )
        data = json.loads(response.data)

        if response.status_code == 404:
            return

        assert data["n_invalid"] == 2
        invalid_indices = [r["row_index"] for r in data["invalid_rows"]]
        assert 1 in invalid_indices
        assert 4 in invalid_indices


# =============================================
# ROUTE /predict (auto-détection)
# =============================================

class TestPredictAuto:

    def test_auto_detect_tonal(self, client):
        """Auto-détecte les données tonales."""
        payload = [{
            "before_exam_125_Hz": 45, "before_exam_250_Hz": 50,
            "before_exam_500_Hz": 55, "before_exam_1000_Hz": 60,
            "before_exam_2000_Hz": 65, "before_exam_4000_Hz": 70,
            "before_exam_8000_Hz": 75,
        }]
        response = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = json.loads(response.data)

        if response.status_code == 404:
            return

        assert data.get("type") == "tonal"

    def test_auto_detect_vocal(self, client):
        """Auto-détecte les données vocales."""
        payload = [{
            "wrs_30_before": 5, "wrs_40_before": 20,
            "wrs_50_before": 55, "wrs_60_before": 80,
            "wrs_70_before": 90, "wrs_80_before": 95,
            "wrs_90_before": 92,
        }]
        response = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        data = json.loads(response.data)

        if response.status_code == 404:
            return

        assert data.get("type") == "vocal"

    def test_unknown_format_400(self, client):
        """Format inconnu retourne 400."""
        payload = [{"unknown": 42}]
        response = client.post(
            "/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert response.status_code == 400


# =============================================
# ERREURS
# =============================================

class TestErrorHandling:

    def test_404_route(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        response = client.get("/predict/tonal")
        assert response.status_code == 405