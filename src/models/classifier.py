import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import settings

logger = logging.getLogger(__name__)


class SentimentClassifier:
    def __init__(self, max_features: int = 50_000, C: float = 1.0):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(C=C, max_iter=1000, n_jobs=-1)),
        ])
        self.classes = ["negative", "positive"]
        self._is_fitted = False

    def train(
        self,
        X_train: List[str],
        y_train: List[int],
        X_val: List[str],
        y_val: List[int],
        experiment_name: str = "sentiment-classification",
    ) -> Dict[str, float]:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            mlflow.log_params({
                "max_features": self.pipeline["tfidf"].max_features,
                "ngram_range": str(self.pipeline["tfidf"].ngram_range),
                "C": self.pipeline["clf"].C,
                "train_size": len(X_train),
                "val_size": len(X_val),
            })

            logger.info("Training model...")
            self.pipeline.fit(X_train, y_train)
            self._is_fitted = True

            metrics = self.evaluate(X_val, y_val)

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                self.pipeline,
                artifact_path="model",
                registered_model_name=settings.MODEL_NAME,
            )

            logger.info(f"Training complete. Metrics: {metrics}")
            return metrics

    def evaluate(self, X: List[str], y: List[int]) -> Dict[str, float]:
        y_pred = self.pipeline.predict(X)
        return {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, average="weighted"), 4),
            "recall": round(recall_score(y, y_pred, average="weighted"), 4),
            "f1": round(f1_score(y, y_pred, average="weighted"), 4),
        }

    def predict(self, text: str) -> Dict:
        if not self._is_fitted:
            raise RuntimeError("Model is not fitted. Call train() or load() first.")
        proba = self.pipeline.predict_proba([text])[0]
        label_idx = int(np.argmax(proba))
        return {
            "label": self.classes[label_idx],
            "confidence": round(float(proba[label_idx]), 4),
            "scores": {
                "negative": round(float(proba[0]), 4),
                "positive": round(float(proba[1]), 4),
            },
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        return [self.predict(t) for t in texts]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.pipeline, f)
        logger.info(f"Model saved to {path}")

    def load(self, path: Path) -> None:
        with open(path, "rb") as f:
            self.pipeline = pickle.load(f)
        self._is_fitted = True
        logger.info(f"Model loaded from {path}")
