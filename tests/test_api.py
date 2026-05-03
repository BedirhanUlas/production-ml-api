import pickle
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_model():
    model = MagicMock()
    model._is_fitted = True
    model.predict.return_value = {
        "label": "positive",
        "confidence": 0.92,
        "scores": {"negative": 0.08, "positive": 0.92},
    }
    return model


@pytest.fixture
def client(mock_model):
    with patch("src.api.main._model", mock_model):
        from src.api.main import app
        with TestClient(app) as c:
            yield c


class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


class TestPredict:
    def test_predict_valid_text(self, client):
        response = client.post("/predict", json={"text": "This movie is absolutely fantastic!"})
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json={"text": "Great product!"}).json()
        assert "label" in data
        assert "confidence" in data
        assert "scores" in data
        assert "text_length" in data

    def test_predict_label_is_valid(self, client):
        data = client.post("/predict", json={"text": "Wonderful experience"}).json()
        assert data["label"] in ["positive", "negative"]

    def test_predict_confidence_is_float(self, client):
        data = client.post("/predict", json={"text": "Excellent!"}).json()
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_predict_empty_text_returns_422(self, client):
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422

    def test_predict_whitespace_only_returns_422(self, client):
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422

    def test_predict_missing_text_returns_422(self, client):
        response = client.post("/predict", json={})
        assert response.status_code == 422

    def test_text_length_matches(self, client):
        text = "This is a test sentence."
        data = client.post("/predict", json={"text": text}).json()
        assert data["text_length"] == len(text)


class TestBatchPredict:
    def test_batch_predict_returns_200(self, client, mock_model):
        mock_model.predict.side_effect = [
            {"label": "positive", "confidence": 0.9, "scores": {"negative": 0.1, "positive": 0.9}},
            {"label": "negative", "confidence": 0.8, "scores": {"negative": 0.8, "positive": 0.2}},
        ]
        response = client.post("/predict/batch", json={"texts": ["Great!", "Terrible!"]})
        assert response.status_code == 200

    def test_batch_count_matches_input(self, client, mock_model):
        mock_model.predict.side_effect = [
            {"label": "positive", "confidence": 0.9, "scores": {"negative": 0.1, "positive": 0.9}},
            {"label": "negative", "confidence": 0.8, "scores": {"negative": 0.8, "positive": 0.2}},
        ]
        data = client.post("/predict/batch", json={"texts": ["Great!", "Terrible!"]}).json()
        assert data["count"] == 2
        assert len(data["results"]) == 2

    def test_batch_empty_list_returns_422(self, client):
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code == 422


class TestStats:
    def test_stats_returns_200(self, client):
        response = client.get("/stats")
        assert response.status_code == 200

    def test_stats_schema(self, client):
        data = client.get("/stats").json()
        assert "total_predictions" in data
        assert "label_distribution" in data
