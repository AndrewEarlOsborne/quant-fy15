# Ethereum Trading System Makefile
# Manages model training, data extraction, and deployment

.PHONY: help setup clean install deploy cloud-deploy status logs
.PHONY: model-build model-up model-down model-logs model-clean model-train model-predict
.PHONY: extraction-build extraction-up extraction-down extraction-logs extraction-clean
.PHONY: extraction-deploy extraction-status extraction-collect extraction-setup
.PHONY: trading-up trading-down trading-logs trading-restart
.PHONY: system-up system-down system-restart system-status system-clean
.PHONY: backup restore test lint format

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE := docker-compose
DOCKER := docker
PROJECT_NAME := ethereum-trading
MODEL_IMAGE := $(PROJECT_NAME)-model
EXTRACTION_IMAGE := $(PROJECT_NAME)-extraction
TRADING_IMAGE := $(PROJECT_NAME)-trading

# Environment variables
ENV_FILE := .env
DATA_DIR := /data
MODELS_DIR := /models
LOGS_DIR := /logs
EXTRACTION_DIR := catalog/extraction-pipeline

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "$(GREEN)Ethereum Trading System Management$(NC)"
	@echo "=================================="
	@echo ""
	@echo "$(YELLOW)Setup Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(setup|install|clean):' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Model Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^model-' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Data Extraction Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^extraction-' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Trading System Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^trading-' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)System Management:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^system-' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Deployment Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(deploy|cloud-deploy|install):' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(YELLOW)Utility Commands:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(status|logs|backup|restore|test|lint|format):' | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install: setup ## Install system service (requires sudo)
	@echo "$(GREEN)Installing system service$(NC)"
	@chmod +x scripts/install.sh
	@sudo ./scripts/install.sh
	@echo "$(GREEN)System service installed$(NC)"

clean: ## Clean up containers, images, and temporary files
	@echo "$(GREEN)Cleaning up system$(NC)"
	@$(DOCKER) system prune -f
	@$(DOCKER) volume prune -f
	@rm -rf $(LOGS_DIR)/*.log
	@rm -rf $(DATA_DIR)/processed/*
	@echo "$(GREEN)Cleanup complete$(NC)"

# Model Commands
model-build: ## Build model training/prediction Docker image
	@echo "$(GREEN)Building model image$(NC)"
	@$(DOCKER) build -t $(MODEL_IMAGE):latest -f Dockerfile.model .
	@echo "$(GREEN)Model image built$(NC)"

model-up: model-build ## Start model training/prediction service
	@echo "$(GREEN)Starting model service$(NC)"
	@$(DOCKER) run -d \
		--name $(PROJECT_NAME)-model \
		--restart unless-stopped \
		--env-file $(ENV_FILE) \
		-v $(PWD)/$(MODELS_DIR):/app/models \
		-v $(PWD)/$(DATA_DIR):/app/data \
		-v $(PWD)/$(LOGS_DIR):/app/logs \
		$(MODEL_IMAGE):latest
	@echo "$(GREEN)Model service started$(NC)"

model-down: ## Stop model service
	@echo "$(GREEN)Stopping model service$(NC)"
	@$(DOCKER) stop $(PROJECT_NAME)-model 2>/dev/null || true
	@$(DOCKER) rm $(PROJECT_NAME)-model 2>/dev/null || true
	@echo "$(GREEN)Model service stopped$(NC)"

model-logs: ## View model service logs
	@$(DOCKER) logs -f $(PROJECT_NAME)-model

model-clean: model-down ## Clean model containers and images
	@echo "$(GREEN)Cleaning model components$(NC)"
	@$(DOCKER) rmi $(MODEL_IMAGE):latest 2>/dev/null || true
	@echo "$(GREEN)Model components cleaned$(NC)"

model-train: ## Run model training
	@echo "$(GREEN)Starting model training$(NC)"
	@$(DOCKER) exec $(PROJECT_NAME)-model python -m src.model.train
	@echo "$(GREEN)Model training complete$(NC)"

model-predict: ## Run model prediction
	@echo "$(GREEN)Running model prediction$(NC)"
	@$(DOCKER) exec $(PROJECT_NAME)-model python -m src.model.predict
	@echo "$(GREEN)Model prediction complete$(NC)"

# Data Extraction Commands (Legacy Docker-based)
extraction-build: ## Build data extraction Docker image
	@echo "$(GREEN)Building extraction image$(NC)"
	@$(DOCKER) build -t $(EXTRACTION_IMAGE):latest -f Dockerfile.extraction .
	@echo "$(GREEN)Extraction image built$(NC)"

extraction-up: extraction-build ## Start data extraction service
	@echo "$(GREEN)Starting extraction service$(NC)"
	@$(DOCKER) run -d \
		--name $(PROJECT_NAME)-extraction \
		--restart unless-stopped \
		--env-file $(ENV_FILE) \
		-v $(PWD)/$(DATA_DIR):/app/data \
		-v $(PWD)/$(LOGS_DIR):/app/logs \
		$(EXTRACTION_IMAGE):latest
	@echo "$(GREEN)Extraction service started$(NC)"

extraction-down: ## Stop data extraction service
	@echo "$(GREEN)Stopping extraction service$(NC)"
	@$(DOCKER) stop $(PROJECT_NAME)-extraction 2>/dev/null || true
	@$(DOCKER) rm $(PROJECT_NAME)-extraction 2>/dev/null || true
	@echo "$(GREEN)Extraction service stopped$(NC)"

extraction-logs: ## View extraction service logs
	@$(DOCKER) logs -f $(PROJECT_NAME)-extraction

extraction-clean: extraction-down ## Clean extraction containers and images
	@echo "$(GREEN)Cleaning extraction components$(NC)"
	@$(DOCKER) rmi $(EXTRACTION_IMAGE):latest 2>/dev/null || true
	@echo "$(GREEN)Extraction components cleaned$(NC)"

# New Cloud-based Extraction Pipeline Commands
extraction-setup: ## Setup extraction pipeline environment
	@echo "$(GREEN)Setting up extraction pipeline$(NC)"
	@cd $(EXTRACTION_DIR) && ./setup.sh
	@echo "$(GREEN)Extraction pipeline setup complete$(NC)"

extraction-deploy: ## Deploy VMs and start extraction
	@echo "$(GREEN)Deploying extraction VMs$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py deploy
	@echo "$(GREEN)Extraction deployment initiated$(NC)"

extraction-status: ## Check extraction pipeline status
	@echo "$(GREEN)Checking extraction status$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py status

extraction-collect: ## Collect extraction results and cleanup VMs
	@echo "$(GREEN)Collecting extraction results$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py collect
	@echo "$(GREEN)Extraction collection complete$(NC)"

extraction-quick-status: ## Quick extraction status check
	@cd $(EXTRACTION_DIR) && python3 main.py status | grep -E "(Completed|Running|Failed)" || echo "No active deployment"

extraction-full-pipeline: extraction-deploy extraction-collect ## Run complete extraction pipeline
	@echo "$(GREEN)Starting full extraction pipeline$(NC)"
	@echo "$(YELLOW)Deployment phase$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py deploy
	@echo "$(YELLOW)Waiting for completion (check status manually)$(NC)"
	@echo "$(YELLOW)Use 'make extraction-status' to monitor progress$(NC)"
	@echo "$(YELLOW)Use 'make extraction-collect' when complete$(NC)"

# Trading System Commands
trading-up: ## Start trading system
	@echo "$(GREEN)Starting trading system$(NC)"
	@python scripts/activate_agent.py &
	@echo "$(GREEN)Trading system started$(NC)"

trading-down: ## Stop trading system
	@echo "$(GREEN)Stopping trading system$(NC)"
	@pkill -f "activate_agent.py" || true
	@echo "$(GREEN)Trading system stopped$(NC)"

trading-logs: ## View trading system logs
	@tail -f $(LOGS_DIR)/trading/trading.log

trading-restart: trading-down trading-up ## Restart trading system

# System Management Commands
system-up: extraction-up model-up trading-up ## Start entire system (extraction + model + trading)
	@echo "$(GREEN)All systems operational$(NC)"

system-down: trading-down model-down extraction-down ## Stop entire system
	@echo "$(GREEN)All systems stopped$(NC)"

system-restart: system-down system-up ## Restart entire system

system-status: ## Check status of all services
	@echo "$(GREEN)System Status$(NC)"
	@echo "================"
	@echo "$(YELLOW)Model Service:$(NC)"
	@$(DOCKER) ps | grep $(PROJECT_NAME)-model || echo "  Not running"
	@echo "$(YELLOW)Extraction Service:$(NC)"
	@$(DOCKER) ps | grep $(PROJECT_NAME)-extraction || echo "  Not running"
	@echo "$(YELLOW)Trading Process:$(NC)"
	@pgrep -f "activate_agent.py" >/dev/null && echo "  Running" || echo "  Not running"
	@echo "$(YELLOW)Cloud Extraction Pipeline:$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py status | grep -E "(deployment|VMs)" | head -3 || echo "  No active deployment"

system-clean: model-clean extraction-clean ## Clean entire system
	@echo "$(GREEN)Complete system cleanup$(NC)"
	@$(DOCKER) system prune -a -f
	@echo "$(GREEN)System cleanup complete$(NC)"

# Deployment Commands
deploy: ## Deploy using Docker locally
	@echo "$(GREEN)Deploying locally$(NC)"
	@chmod +x scripts/deploy.sh
	@./scripts/deploy.sh
	@echo "$(GREEN)Local deployment complete$(NC)"

cloud-deploy: ## Deploy to cloud instance
	@echo "$(GREEN)Deploying to cloud$(NC)"
	@chmod +x scripts/cloud_deploy.sh
	@./scripts/cloud_deploy.sh
	@echo "$(GREEN)Cloud deployment complete$(NC)"

# Utility Commands
status: system-status ## Alias for system-status

logs: ## View aggregated logs from all services
	@echo "$(GREEN)Viewing system logs$(NC)"
	@echo "Press Ctrl+C to exit"
	@tail -f $(LOGS_DIR)/*/*.log 2>/dev/null || echo "No log files found"

backup: ## Create backup of data and models
	@echo "$(GREEN)Creating backup$(NC)"
	@mkdir -p $(BACKUP_DIR)
	@tar -czf $(BACKUP_DIR)/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		$(DATA_DIR) $(MODELS_DIR) $(ENV_FILE) $(EXTRACTION_DIR)/.env 2>/dev/null || true
	@echo "$(GREEN)Backup created in $(BACKUP_DIR)$(NC)"

restore: ## Restore from latest backup (interactive)
	@echo "$(GREEN)Available backups:$(NC)"
	@ls -la $(BACKUP_DIR)/*.tar.gz 2>/dev/null || echo "No backups found"
	@echo "To restore: tar -xzf $(BACKUP_DIR)/backup-YYYYMMDD-HHMMSS.tar.gz"

test: ## Run tests
	@echo "$(GREEN)Running tests$(NC)"
	@python -m pytest tests/ -v || echo "$(RED)Tests failed$(NC)"

lint: ## Run linting
	@echo "$(GREEN)Running linting$(NC)"
	@python -m flake8 src/ scripts/ || echo "$(YELLOW)Linting issues found$(NC)"

format: ## Format code
	@echo "$(GREEN)Formatting code$(NC)"
	@python -m black src/ scripts/
	@echo "$(GREEN)Code formatted$(NC)"

# Development shortcuts
dev-up: extraction-up model-up ## Start development environment (no trading)
dev-down: model-down extraction-down ## Stop development environment
dev-restart: dev-down dev-up ## Restart development environment

# Cloud extraction development shortcuts
dev-extraction-setup: extraction-setup ## Setup extraction pipeline for development
dev-extraction-test: ## Test extraction pipeline with small deployment
	@echo "$(GREEN)Testing extraction pipeline$(NC)"
	@echo "$(YELLOW)Ensure .env is configured for small test$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py deploy

# Quick status checks
quick-status: ## Quick status check
	@echo "$(GREEN)Quick Status$(NC)"
	@$(DOCKER) ps --format "table {{.Names}}\t{{.Status}}" | grep $(PROJECT_NAME) || echo "No services running"

# Emergency stop
emergency-stop: ## Emergency stop all services
	@echo "$(RED)EMERGENCY STOP$(NC)"
	@$(DOCKER) stop $(shell $(DOCKER) ps -q --filter "name=$(PROJECT_NAME)") 2>/dev/null || true
	@pkill -f "activate_agent.py" || true
	@echo "$(RED)All services force stopped$(NC)"

# Resource monitoring
monitor: ## Monitor system resources
	@echo "$(GREEN)System Resources$(NC)"
	@echo "==================="
	@$(DOCKER) stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | grep $(PROJECT_NAME) || echo "No containers running"