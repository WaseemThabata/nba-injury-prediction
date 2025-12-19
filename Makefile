# Makefile for NBA Injury Prediction Project

.PHONY: help install install-dev test lint format clean run docker-build docker-run

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install production dependencies
	pip install -r requirements.txt

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pre-commit install

test:  ## Run tests with pytest
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

test-quick:  ## Run tests without coverage
	pytest tests/ -v

lint:  ## Run linters (flake8, mypy)
	flake8 src scripts app.py --max-line-length=100
	mypy src --ignore-missing-imports

format:  ## Format code with black
	black src scripts app.py tests

format-check:  ## Check code formatting
	black --check src scripts app.py tests

clean:  ## Clean generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage coverage.xml

run-pipeline:  ## Run the training pipeline
	python scripts/run_pipeline.py

run-demo:  ## Run the synthetic data demo
	python scripts/demo_synthetic.py

run-app:  ## Run the Streamlit web app
	streamlit run app.py

docker-build:  ## Build Docker image
	docker build -t nba-injury-prediction:latest .

docker-run:  ## Run Docker container
	docker-compose up -d

docker-stop:  ## Stop Docker container
	docker-compose down

docker-logs:  ## View Docker logs
	docker-compose logs -f

build:  ## Build Python package
	python -m build

publish-test:  ## Publish to TestPyPI
	python -m twine upload --repository testpypi dist/*

publish:  ## Publish to PyPI
	python -m twine upload dist/*

.DEFAULT_GOAL := help
