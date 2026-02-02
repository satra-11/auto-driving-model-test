# ============================================
# Makefile for LGTCN Project
# ============================================
PYTHON := $(shell command -v uv > /dev/null 2>&1 && echo "uv run python" || echo "python3")
.PHONY: help install install-dev sync lint test mlflow clean extract train evaluate all

# デフォルトはヘルプを表示
help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install dependencies using uv"
	@echo "  install-dev   Install dev dependencies using uv"
	@echo "  sync          Sync dependencies using uv"
	@echo ""
	@echo "Development:"
	@echo "  lint          Run linting (ruff, mypy)"
	@echo "  test          Run tests with pytest"
	@echo ""
	@echo "Training & Evaluation:"
	@echo "  extract       Extract features from raw images"
	@echo "  train         Train the driving models"
	@echo "  evaluate      Evaluate trained models"
	@echo "  compare       Compare robustness (LTCN vs NODE)"
	@echo ""
	@echo "Tools:"
	@echo "  mlflow        Start MLflow UI"
	@echo ""
	@echo "Utility:"
	@echo "  clean         Remove cache files"
	@echo "  all           Run full pipeline (train → evaluate)"

# ============================================
# Setup
# ============================================

install:
	uv sync --no-dev

install-dev:
	uv sync

sync:
	uv sync

# ============================================
# Development
# ============================================

lint:
	uv run ruff check src tests scripts
	uv run mypy src

test:
	uv run pytest

# ============================================
# Training & Evaluation
# ============================================

extract:
	$(PYTHON) scripts/extract_features.py

train:
	$(PYTHON) scripts/train_driving.py --model ltcn
	$(PYTHON) scripts/train_driving.py --model node

evaluate:
	$(PYTHON) scripts/evaluate_corruption_robustness.py --model-type ltcn node --corruption-type noise   --levels 0.1,0.2,0.3,0.4,0.5
	$(PYTHON) scripts/evaluate_corruption_robustness.py --model-type ltcn node --corruption-type overexposure --levels 0.1,0.2,0.3,0.4,0.5



# ============================================
# Tools
# ============================================

mlflow:
	uv run mlflow ui --port 5001

# ============================================
# Utility
# ============================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

all: train evaluate
