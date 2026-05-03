"""
Training script for the sentiment classifier.
Uses the IMDB dataset from HuggingFace datasets (no API key required).
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from src.data.preprocessing import batch_preprocess
from src.models.classifier import SentimentClassifier
from src.config import settings, MODEL_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading IMDB dataset...")
    dataset = load_dataset("imdb")

    train_texts = dataset["train"]["text"]
    train_labels = dataset["train"]["label"]
    test_texts = dataset["test"]["text"]
    test_labels = dataset["test"]["label"]

    logger.info(f"Train size: {len(train_texts):,} | Test size: {len(test_texts):,}")

    logger.info("Preprocessing text...")
    X_train = batch_preprocess(train_texts)
    X_test = batch_preprocess(test_texts)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, train_labels, test_size=0.1, random_state=42, stratify=train_labels
    )

    logger.info("Training model with MLflow tracking...")
    classifier = SentimentClassifier(max_features=50_000, C=1.0)
    metrics = classifier.train(X_train, y_train, X_val, y_val)

    logger.info(f"Validation metrics: {metrics}")

    model_path = MODEL_DIR / "sentiment_model.pkl"
    classifier.save(model_path)

    logger.info("Evaluating on test set...")
    test_metrics = classifier.evaluate(X_test, test_labels)
    logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    main()
