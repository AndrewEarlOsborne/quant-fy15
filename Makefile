# Ethereum Trading System Makefile
# Manages model training, data extraction, and deployment

.PHONY: help setup clean install deploy cloud-deploy status logs
.PHONY: model-build model-up model-down model-logs model-clean model-train model-predict
.PHONY: extractor-build extractor-up extractor-down extractor-logs extractor-clean
.PHONY: extractor-deploy extractor-status extractor-collect extractor-setup
.PHONY: trader-up trader-down trader-logs trader-restart
.PHONY: system-up system-down system-restart system-status system-clean
.PHONY: backup restore test lint format

# Default target
.DEFAULT_GOAL := help

# Variables
DOCKER_COMPOSE := docker-compose
DOCKER := docker
PROJECT_NAME := ethereum-trading
EXTRACTOR_IMAGE := extractor
TRADER_IMAGE := trader
DOCKER_DIR := docker

# Environment variables
ENV_FILE := .env
DATA_DIR := data
MODELS_DIR := models
LOGS_DIR := logs
EXTRACTION_DIR := catalog/extractor-pipeline
EXTRACTOR_COMPOSE := $(DOCKER_DIR)/extractor-compose.yaml
TRADER_COMPOSE := $(DOCKER_DIR)/trader-compose.yaml

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# Help target
help: ## Show this help message
	@echo "Ethereum Trading System Management"
	@echo "Setup Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(setup|install|clean):' | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo "Extractor Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^extractor-' | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo "Trader Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^trader-' | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo "System Management:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^system-' | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo "Utility Commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -E '^(status|logs|backup|restore|test|lint|format):' | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

clean: ## Clean up containers, images, temporary files, and cloud VMs
	@echo "$(YELLOW)Cleaning up local containers and images...$(NC)"
	@$(DOCKER) stop $$($(DOCKER) ps -aq) 2>/dev/null || true
	@$(DOCKER) rm $$($(DOCKER) ps -aq) 2>/dev/null || true
	@$(DOCKER) rmi $$($(DOCKER) images -q) 2>/dev/null || true
	@$(DOCKER) system prune -a -f
	@$(DOCKER) volume prune -f
	@$(DOCKER) builder prune -a -f
	@rm -rf $(LOGS_DIR)/*.log
	@rm -rf $(DATA_DIR)/processed/*
	@echo "$(YELLOW)Cleaning up cloud extraction VMs...$(NC)"
	@cd $(EXTRACTION_DIR) && chmod +x cleanup.sh && ./cleanup.sh --force || echo "$(RED)Cloud cleanup failed or no VMs to clean$(NC)"
	@echo "$(GREEN)Cleanup completed$(NC)"


# Extractor Commands
extractor-build: ## Build extractor Docker image
	@$(DOCKER) build --no-cache --pull -t $(EXTRACTOR_IMAGE):latest -f $(DOCKER_DIR)/Dockerfile.extractor .

extractor-up: ## Start extractor service
	@$(DOCKER_COMPOSE) -f $(EXTRACTOR_COMPOSE) up -d

extractor-down: ## Stop extractor service
	@$(DOCKER_COMPOSE) -f $(EXTRACTOR_COMPOSE) down

extractor-logs: ## View extractor service logs
	@$(DOCKER_COMPOSE) -f $(EXTRACTOR_COMPOSE) logs -f

extractor-clean: extractor-down ## Clean extractor containers and images
	@$(DOCKER) rmi $(EXTRACTOR_IMAGE):latest 2>/dev/null || true

# Cloud Extraction Pipeline Commands
extractor-deploy: ## Deploy extraction VMs to cloud
	@echo "$(YELLOW)Deploying extraction pipeline to cloud...$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py deploy

extractor-status: ## Check status of cloud extraction VMs
	@echo "$(YELLOW)Checking extraction pipeline status...$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py status

extractor-collect: ## Collect results from extraction VMs
	@echo "$(YELLOW)Collecting results from extraction VMs...$(NC)"
	@cd $(EXTRACTION_DIR) && python3 main.py collect

extractor-monitor: ## Monitor screen sessions on extraction VMs
	@echo "$(YELLOW)Monitoring screen sessions on extraction VMs...$(NC)"
	@cd $(EXTRACTION_DIR) && chmod +x monitor_screens.sh && ./monitor_screens.sh --all

extractor-restart-screens: ## Restart failed screen sessions on VMs
	@echo "$(YELLOW)Restarting failed screen sessions...$(NC)"
	@cd $(EXTRACTION_DIR) && chmod +x monitor_screens.sh && ./monitor_screens.sh --restart

extractor-cleanup: ## Emergency cleanup of all extraction VMs
	@echo "$(RED)Emergency cleanup of extraction VMs...$(NC)"
	@cd $(EXTRACTION_DIR) && chmod +x cleanup.sh && ./cleanup.sh --force


# Trader Commands
trader-build: ## Build trader Docker image
	@$(DOCKER) build --no-cache --pull -t $(TRADER_IMAGE):latest -f $(DOCKER_DIR)/Dockerfile.trader .

trader-up: ## Start trader service
	@$(DOCKER_COMPOSE) -f $(TRADER_COMPOSE) up -d

trader-down: ## Stop trader service
	@$(DOCKER_COMPOSE) -f $(TRADER_COMPOSE) down

trader-logs: ## View trader service logs
	@$(DOCKER_COMPOSE) -f $(TRADER_COMPOSE) logs -f

trader-clean: trader-down ## Clean trader containers and images
	@$(DOCKER) rmi $(TRADER_IMAGE):latest 2>/dev/null || true


trader-restart: trader-down trader-up ## Restart trader service

# System Management Commands
system-up: extractor-up trader-up ## Start entire system

system-down: trader-down extractor-down ## Stop entire system


system-restart: system-down system-up ## Restart entire system

system-status: ## Check status of all services
	@$(DOCKER) ps | grep extractor || echo "Extractor: Not running"
	@$(DOCKER) ps | grep trader || echo "Trader: Not running"

system-clean: trader-clean extractor-clean ## Clean entire system
	@$(DOCKER) system prune -a -f

# Utility Commands
status: system-status ## Alias for system-status

logs: ## View aggregated logs from all services
	@tail -f $(LOGS_DIR)/*/*.log 2>/dev/null || echo "No log files found"

backup: ## Create backup of data and models
	@mkdir -p $(BACKUP_DIR)
	@tar -czf $(BACKUP_DIR)/backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		$(DATA_DIR) $(MODELS_DIR) $(ENV_FILE) 2>/dev/null || true

restore: ## Restore from latest backup
	@ls -la $(BACKUP_DIR)/*.tar.gz 2>/dev/null || echo "No backups found"
	@echo "To restore: tar -xzf $(BACKUP_DIR)/backup-YYYYMMDD-HHMMSS.tar.gz"

test: ## Run tests
	@python -m pytest tests/ -v

lint: ## Run linting
	@python -m flake8 src/ scripts/

format: ## Format code
	@python -m black src/ scripts/