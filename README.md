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

The Ethereum Trading System uses a **two-container architecture** optimized for performance and simplicity:

1. **Extractor Container** - Pre-training data extraction with GCC-optimized processing
2. **Trader Container** - Unified model training, live extraction, and automated trading

### Key Design Principles

- **Two-Stage Processing**: Pre-training extraction + unified trading operations
- **Performance Optimization**: GCC-compiled components for high-speed data processing
- **Containerization**: Docker-based deployment for consistent environments
- **File-based Storage**: No external databases for simplified deployment
- **Unified Architecture**: Model training, live extraction, and trading in single container

## 🔧 Architecture Components

### 1. Extractor Container (`docker/Dockerfile.extractor`)

**GCC-Optimized Processing:**
- High-performance C/C++ components for blockchain data extraction
- Historical data collection and preprocessing
- Feature engineering with technical indicators
- Optimized data structures for memory efficiency

**Pre-training Data Pipeline:**
- Raw blockchain data ingestion and validation
- Technical indicator calculation (moving averages, RSI, MACD)
- Whale transaction analysis and aggregation
- Validator performance metrics extraction

**Data Flow:**
```
Raw Blockchain Data → GCC Processing → Feature Engineering → Clean Dataset
        ↓                   ↓               ↓                    ↓
[Historical Data]    [C/C++ Optimized]  [Technical Indicators] [CSV Output]
```

### 2. Trader Container (`docker/Dockerfile.trader`)

**Unified Operations:**
- `model-development/validator_model.py` - Stacking ensemble ML model (TCN + Transformer + XGBoost)
- `catalog/src/trade_engine.py` - Exchange integration and order execution
- `catalog/src/trading_scheduler.py` - Live data extraction and trading automation
- `model-development/utils.py` - Feature engineering utilities

**Integrated Pipeline:**
```
Live Data → Model Training → Prediction → Risk Management → Trading
    ↓           ↓              ↓            ↓               ↓
[Real-time]  [Ensemble ML]   [3-class]    [Position Size] [Exchange API]
```

**Trading Components (within Trader Container):**
- `data_manager.py` - File-based data storage and management
- `config.py` - System configuration management
- Live extraction for real-time trading decisions
- Automated risk management and position sizing

**Unified Trading Flow:**
```
Live Extraction → Model Inference → Risk Check → Execute Trade
       ↓               ↓            ↓           ↓
[Real-time Data] [Latest Model] [Position Size] [Exchange API]
```

## 📁 Project Structure

```
Ethereum-Validator-Trader/
├── Makefile                          # Central control interface
├── catalog/
│   ├── extractor-pipeline/          # Cloud extraction orchestration
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

### 1. Pre-Training Extraction Phase (GCC-Optimized)
```
Extractor Container → GCC Processing → Feature Engineering → Dataset Output
         ↓                   ↓               ↓                    ↓
[Raw Blockchain]    [C/C++ Optimized]  [Technical Indicators] [Clean CSV]
         ↓                   ↓               ↓                    ↓
[Historical Data]   [High Performance]  [Whale Analysis]       [Model Ready]
```

### 2. Model Training Phase
```
Aggregated Data → Feature Engineering → Model Training → Model Persistence
       ↓                 ↓                   ↓               ↓
[CSV Collection]  [Technical Indicators] [Ensemble ML]  [Saved Models]
       ↓                 ↓                   ↓               ↓
[Price+Whale+Val] [Lag Features+Stats] [TCN+Transform+XGB] [model/ dir]
```

### 3. Live Trading Phase (Unified Container)
```
Live Extraction → Model Training → Prediction → Risk Check → Trade Execution
       ↓              ↓             ↓            ↓              ↓
[Real-time Data] [Ensemble ML]   [3-class]   [Position Size] [Exchange API]
       ↓              ↓             ↓            ↓              ↓
[Feature Eng]    [TCN+Trans+XGB] [Confidence] [Risk Limits]   [Order Placed]
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

### Container Management
```bash
make extractor-up        # Start pre-training extraction container
make trader-up          # Start unified trading container
make system-up          # Start both containers
```

### Trading Operations (Unified)
```bash
make trader-up          # Start trading container (includes model training)
make trader-down        # Stop trading container
make trader-logs        # View trading logs
```

## 🛠️ Setup and Deployment

### Prerequisites
```bash
# System requirements
- Docker 20.10+
- GCC compiler (for optimized extraction)
- Python 3.8+
- 8GB RAM minimum
- 20GB disk space
```

### Quick Start
1. **Environment Setup**
```bash
# Clone repository
git clone <repository-url> ethereum-trader-system
cd ethereum-trader-system

# Configure environment
cp template.env .env
# Edit .env with your API keys and configuration
```

2. **Initialize System**
```bash
# Setup data directories
make setup

# Start pre-training extraction
make extractor-up

# Start trading system (includes model training and live trading)
make trader-up
```

### Container Configuration
```bash
# Configure .env for extraction and trading
ETHEREUM_PROVIDER_URLS=url1,url2,url3
BINANCE_API_KEY=your-api-key
BINANCE_SECRET=your-secret
START_DATE=2024-01-01-00:00
END_DATE=2024-12-31-23:59
GCC_OPTIMIZATION_FLAGS=-O3 -march=native
```

## 🔄 Development Workflow

### 1. Pre-Training Data Extraction
```bash
# Start GCC-optimized extraction
make extractor-up

# Monitor extraction logs
make extractor-logs

# Stop when data collection complete
make extractor-down
```

### 2. Trading Operations
```bash
# Start unified trading container
make trader-up

# View trading logs (includes model training)
make trader-logs

# Monitor system status
make system-status
```

### 3. System Management
```bash
# Start both containers
make system-up

# Monitor system status
make system-status

# View aggregated logs
make logs

# Clean up system
make clean

# Backup data and models
make backup
```

## 📊 Monitoring and Observability

### System Health Checks
```bash
make system-status        # Overall system health
make extractor-logs      # Pre-training extraction logs
make trader-logs        # Trading execution logs
```

### Performance Monitoring
- Model accuracy and F1 score tracking
- Trading performance metrics
- GCC-optimized extraction performance
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

### Production Deployment
```bash
# Build and start both containers
make system-up

# Enable monitoring alerts
make monitor

# View system resource usage
make monitor
```

### Performance Considerations
- GCC optimization flags for extraction performance
- Container resource allocation and limits
- Model retraining frequency adjustment
- Trading frequency and position sizing optimization

This streamlined two-container architecture provides an optimized solution for automated Ethereum trading with GCC-enhanced performance and unified operational capabilities.