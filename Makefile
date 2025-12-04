.DEFAULT_GOAL := help
.PHONY: help

DOCKER_COMPOSE := docker-compose
PYTHON := python3

help:
	@echo "Ethereum Validator Trader - Available Commands"
	@echo "================================================"
	@echo ""
	@echo "Development Commands:"
	@echo "  make dev-extractor        Run extractor pipeline locally"
	@echo "  make dev-model            Run model training locally"
	@echo "  make lint                 Run ruff linter on model-builder code"
	@echo ""
	@echo "Data Management:"
	@echo "  make clean-extractor      Clean extractor state files"
	@echo "  make clean-model          Clean model files"
	@echo ""

clean-extractor:
	@echo "Cleaning extractor state files..."
	@rm -f logs/vm_orchestrator_state.json
	@echo "Extractor state cleaned!"

clean-model:
	@echo "Cleaning model files..."
	@rm -rf models/*
	@echo "Model files cleaned!"

dev-extractor:
	@echo "Running extractor pipeline locally..."
	@$(PYTHON) catalog/extractor-pipeline/main.py

dev-model:
	@echo "Running model training locally..."
	@$(PYTHON) catalog/model-builder/test_pipeline.py

dev-trader:
	@echo "Running trader agent locally..."
	@$(PYTHON) catalog/src/activate_agent.py

lint:
	@echo "Running ruff linter on model-builder..."
	@if $(PYTHON) -m ruff --version > /dev/null 2>&1; then \
		$(PYTHON) -m ruff check catalog/model-builder/ --select E,F,W,N --fix; \
		echo "Linting complete!"; \
	else \
		echo "Warning: ruff not installed. Install with: pip install ruff"; \
		echo "Skipping linting..."; \
	fi

extractor-build:
	@echo "Building extractor Docker image..."
	@docker build -f docker/Dockerfile.extractor -t eth-trader-extractor:latest .

extractor-up: extractor-build
	@echo "Starting extractor container..."
	@docker run -d --name eth-extractor \
		--restart unless-stopped \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/data:/app/data \
		eth-trader-extractor:latest
	@echo "Extractor container started!"

extractor-down:
	@echo "Stopping extractor container..."
	@docker stop eth-extractor 2>/dev/null || true
	@docker rm eth-extractor 2>/dev/null || true
	@echo "Extractor container stopped!"

extractor-logs:
	@docker logs -f eth-extractor

extractor-shell:
	@docker exec -it eth-extractor /bin/bash

extractor-restart: extractor-down extractor-up

trader-build:
	@echo "Building trader Docker image..."
	@docker build -f docker/Dockerfile.trader -t eth-trader-agent:latest .

trader-up: trader-build
	@echo "Starting trader container..."
	@docker run -d --name eth-trader \
		--restart unless-stopped \
		-v $(PWD)/logs:/app/logs \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/data:/app/data \
		-p 8000:8000 \
		eth-trader-agent:latest
	@echo "Trader container started!"

trader-down:
	@echo "Stopping trader container..."
	@docker stop eth-trader 2>/dev/null || true
	@docker rm eth-trader 2>/dev/null || true
	@echo "Trader container stopped!"

trader-logs:
	@docker logs -f eth-trader

trader-shell:
	@docker exec -it eth-trader /bin/bash

trader-restart: trader-down trader-up

system-up: extractor-up trader-up
	@echo "All systems operational!"

system-down: extractor-down trader-down
	@echo "All systems stopped!"

system-restart: system-down system-up

system-status:
	@echo "System Status:"
	@echo "=============="
	@docker ps -a --filter "name=eth-" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

system-clean: system-down
	@echo "Cleaning all Docker resources..."
	@docker system prune -f
	@echo "System cleaned!"

logs:
	@echo "Container Logs:"
	@echo "==============="
	@docker logs eth-extractor --tail=50 2>/dev/null || echo "Extractor not running"
	@echo ""
	@docker logs eth-trader --tail=50 2>/dev/null || echo "Trader not running"

monitor:
	@echo "System Resource Monitor:"
	@echo "========================"
	@docker stats eth-extractor eth-trader

ps:
	@docker ps --filter "name=eth-"

stats:
	@docker stats --no-stream eth-extractor eth-trader 2>/dev/null || echo "Containers not running"

backup:
	@echo "Creating backup..."
	@mkdir -p backups
	@tar -czf backups/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz models/ data/ logs/
	@echo "Backup created in backups/"

restore:
	@echo "Available backups:"
	@ls -1 backups/*.tar.gz 2>/dev/null || echo "No backups found"
	@echo "To restore, run: tar -xzf backups/<backup-file>.tar.gz"

test:
	@echo "Running all tests..."
	@$(PYTHON) -m pytest tests/ -v

test-unit:
	@echo "Running unit tests..."
	@$(PYTHON) -m pytest tests/unit/ -v

test-integration:
	@echo "Running integration tests..."
	@$(PYTHON) -m pytest tests/integration/ -v

emergency-stop:
	@echo "EMERGENCY STOP: Halting all trading operations..."
	@docker stop eth-trader 2>/dev/null || true
	@echo "Trading stopped! Manual intervention required to restart."

kill-all:
	@echo "Force killing all containers..."
	@docker kill eth-extractor eth-trader 2>/dev/null || true
	@docker rm eth-extractor eth-trader 2>/dev/null || true
	@echo "All containers killed!"

.PHONY: help dev-extractor dev-model clean-extractor clean-model