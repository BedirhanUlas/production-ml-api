.PHONY: install train test lint docker-up docker-down run

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

test:
	pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

run:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down

mlflow-ui:
	mlflow ui --port 5000
