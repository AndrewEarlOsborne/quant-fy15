# Ethereum Trading System - Technical Architecture

A comprehensive machine learning-based automated trading system for Ethereum with modular data extraction, model development, and trading execution components.

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Architecture Components](#architecture-components)
- [Project Structure](#project-structure)
- [Data Flow](#data-flow)
- [Setup and Deployment](#setup-and-deployment)
- [Makefile Controller](#makefile-controller)
- [Development Workflow](#development-workflow)

## 🏗️ System Overview

The Ethereum Trading System consists of three main architectural components:

1. **Cloud Extraction Pipeline** - Scalable VM-based data collection
2. **ML Model Development** - Feature engineering and ensemble modeling
3. **Trading Control Plane** - Automated execution and portfolio management

### Key Design Principles

- **Modularity**: Clear separation between extraction, modeling, and trading
- **Scalability**: Cloud-based extraction using multiple VMs
- **Containerization**: Docker-based deployment for consistent environments
- **File-based Storage**: No external databases for simplified deployment
- **Fault Tolerance**: Graceful error handling and recovery mechanisms

## 🔧 Architecture Components

### 1. Cloud Extraction Pipeline (`catalog/extraction-pipeline/`)

**Local Orchestration Plane:**
- `main.py` - CLI interface for VM deployment and management
- `orchestrator.py` - Core VM orchestration and data collection logic
- Manages VM lifecycle: deploy → monitor → collect → cleanup

**VM Extraction Workers:**
- Automatically cloned minimal extraction codebase on each VM
- Independent operation with time-partitioned data extraction
- Fault-tolerant processing with status reporting

**Data Flow:**
```
Local Machine → GCP VMs → Parallel Extraction → Data Aggregation → Local Storage
     ↓              ↓             ↓                ↓                    ↓
[Deploy Command] [VM Fleet] [Time Windows] [CSV Collection] [catalog/data/]
```

### 2. Model Development (`model-development/`)

**Core Components:**
- `validator_model.py` - Stacking ensemble ML model (TCN + Transformer + XGBoost)
- `main.py` - Model training and evaluation pipeline
- `utils.py` - Feature engineering utilities

**Model Architecture:**
```
Raw Data → Feature Engineering → Ensemble Training → Prediction
    ↓             ↓                    ↓               ↓
[Price+Whale+  [Technical        [TCN+Transform+   [3-class
 Validator]     Indicators]        XGB+Meta]        prediction]
```

### 3. Trading Control Plane (`catalog/src/`)

**Control Components:**
- `trade_engine.py` - Exchange integration and order execution
- `trading_scheduler.py` - Automated prediction and trading schedule
- `data_manager.py` - File-based data storage and management
- `config.py` - System configuration management

**Trading Flow:**
```
Scheduled Trigger → Load Model → Make Prediction → Risk Check → Execute Trade
        ↓               ↓            ↓             ↓           ↓
[Daily 23:00 UTC] [Load Latest] [Confidence] [Position Size] [Exchange API]
```

## 📁 Project Structure

```
Ethereum-Validator-Trader/
├── Makefile                          # Central control interface
├── catalog/
│   ├── extraction-pipeline/          # Cloud extraction orchestration
│   │   ├── main.py                   # CLI for VM deployment
│   │   ├── orchestrator.py           # VM management logic
│   │   └── readme                    # Pipeline architecture docs
│   ├── src/                          # Trading control plane
│   │   ├── config.py                 # System configuration
│   │   ├── trade_engine.py           # Trading execution
│   │   ├── trading_scheduler.py      # Automated scheduling
│   │   ├── data_manager.py           # Data management
│   │   └── activate_agent.py         # Main trading agent
│   └── scripts/                      # Deployment scripts
│       ├── deploy.sh                 # Local deployment
│       ├── quickstart-local.sh       # Local development setup
│       └── quikstart-cloud.sh        # Cloud deployment
├── model-development/                 # ML model development
│   ├── validator_model.py            # Ensemble model implementation
│   ├── main.py                       # Training pipeline
│   └── utils.py                      # Feature engineering
├── data/                             # Data storage directory
├── docker/                           # Container configuration
│   ├── Dockerfile                    # Main application container
│   ├── docker-compose.yaml           # Multi-service orchestration
│   └── requirements.txt              # Python dependencies
└── file_aggregator.py               # Data aggregation utility
```

## 🌊 Data Flow

### 1. Extraction Phase (Cloud-based)
```
Orchestrator → VM Fleet → Parallel Processing → Data Collection
     ↓              ↓             ↓                    ↓
[main.py deploy] [GCP VMs]  [Time Windows]    [CSV Files]
                     ↓             ↓                    ↓
              [Clone Repo]  [Extract Data]      [Status Updates]
                     ↓             ↓                    ↓
              [Run Scripts] [Independent]       [Completion Flag]
```

### 2. Model Training Phase
```
Aggregated Data → Feature Engineering → Model Training → Model Persistence
       ↓                 ↓                   ↓               ↓
[CSV Collection]  [Technical Indicators] [Ensemble ML]  [Saved Models]
       ↓                 ↓                   ↓               ↓
[Price+Whale+Val] [Lag Features+Stats] [TCN+Transform+XGB] [model/ dir]
```

### 3. Trading Phase (Automated)
```
Schedule Trigger → Load Model → Feature Prep → Prediction → Trade Execution
       ↓              ↓           ↓            ↓              ↓
[Daily 23:00]   [Latest Model] [Recent Data] [Classification] [Exchange API]
       ↓              ↓           ↓            ↓              ↓
[Cron Job]      [Load Weights] [14-day Win]  [0,1,2 Class]  [Order Placed]
```

## 🚀 Makefile Controller

The Makefile serves as the central control interface for the entire system:

### System Management
```bash
make help              # Show all available commands
make system-up         # Start entire system (extraction + model + trading)
make system-down       # Stop entire system
make system-status     # Check status of all services
```

### Extraction Pipeline
```bash
make extraction-deploy    # Deploy VMs and start cloud extraction
make extraction-status    # Check VM status and progress
make extraction-collect   # Collect results and cleanup VMs
```

### Model Operations
```bash
make model-train          # Train new model version
make model-predict        # Run prediction with current model
make model-up            # Start model service container
```

### Trading Operations
```bash
make trading-up          # Start automated trading agent
make trading-down        # Stop trading agent
make trading-logs        # View trading logs
```

## 🛠️ Setup and Deployment

### Prerequisites
```bash
# System requirements
- Docker 20.10+
- Google Cloud SDK (for extraction pipeline)
- Python 3.8+
- 4GB RAM minimum
- 20GB disk space
```

### Quick Start
1. **Environment Setup**
```bash
# Clone repository
git clone <repository-url> ethereum-trading-system
cd ethereum-trading-system

# Configure environment
cp template.env .env
# Edit .env with your API keys and configuration
```

2. **Initialize System**
```bash
# Setup data directories
make setup

# Deploy extraction pipeline
make extraction-deploy

# Train initial model (after data collection)
make model-train

# Start trading system
make trading-up
```

### Cloud Extraction Configuration
```bash
# Configure .env for extraction pipeline
GCP_PROJECT_ID=your-project-id
EXTRACTION_REPO=https://github.com/your-org/extraction-repo
START_DATE=2024-01-01-00:00
END_DATE=2024-12-31-23:59
NUM_VMS=10
ETHEREUM_PROVIDER_URLS=url1,url2,url3
```

## 🔄 Development Workflow

### 1. Data Collection
```bash
# Deploy extraction VMs
make extraction-deploy

# Monitor progress
make extraction-status

# Collect results when complete
make extraction-collect
```

### 2. Model Development
```bash
# Start development environment
make dev-up

# Train new model version
make model-train

# Evaluate model performance
make model-predict
```

### 3. Trading Deployment
```bash
# Start trading system
make system-up

# Monitor system status
make system-status

# View logs
make logs
```

### 4. Maintenance
```bash
# Clean up system
make clean

# Backup data and models
make backup

# Update model with new data
make model-retrain
```

## 📊 Monitoring and Observability

### System Health Checks
```bash
make system-status        # Overall system health
make extraction-status    # VM extraction progress
make model-logs          # Model training/prediction logs
make trading-logs        # Trading execution logs
```

### Performance Monitoring
- Model accuracy and F1 score tracking
- Trading performance metrics
- VM extraction completion rates
- Exchange API latency monitoring

## 🔒 Security and Risk Management

### Trading Risk Controls
- Maximum position size limits (10% of portfolio)
- Risk per trade limits (2% per trade)
- Confidence threshold requirements (60% minimum)
- Sandbox mode for testing

### API Security
- Exchange API keys stored in environment variables
- Sandbox mode for development and testing
- Rate limiting and error handling

## 🚀 Production Deployment

### Cloud Deployment
```bash
# Deploy to cloud with full monitoring
make cloud-deploy

# Production system startup
make system-up

# Enable monitoring alerts
make monitor
```

### Scaling Considerations
- Increase VM count for larger data extraction
- Model retraining frequency adjustment
- Trading frequency and position sizing optimization

This architecture provides a comprehensive, scalable solution for automated Ethereum trading with clear separation of concerns and robust operational capabilities.