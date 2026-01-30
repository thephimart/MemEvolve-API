# MemEvolve Scripts

This directory contains scripts for development, deployment, maintenance, and testing of MemEvolve.

## 🚀 Core Scripts

### setup.sh
**Purpose**: Interactive installer for MemEvolve API proxy
**Usage**: `./setup.sh`
**Features**:
- Creates virtual environment
- Configures API endpoints and storage backends
- Sets up startup scripts and aliases
- Supports JSON/vector/graph storage backends

### uninstall.sh
**Purpose**: Linux uninstaller for MemEvolve
**Usage**: `./uninstall.sh`
**Features**:
- Removes startup scripts and aliases
- Cleans up PATH modifications
- Preserves user data and config

### start_api.py
**Purpose**: Start the MemEvolve API proxy server
**Usage**: `python scripts/start_api.py` or `python scripts/start_api.py --reload`
**Features**:
- FastAPI-based API server with comprehensive business analytics
- Transparent proxy to OpenAI-compatible endpoints
- Automatic memory injection with quality scoring
- Intelligent auto-evolution with multiple triggers
- Health monitoring and executive-level business metrics
- Real-time ROI tracking and trend analysis

## 🔧 Development Tools

### format.sh
**Purpose**: Code formatting using autopep8
**Usage**: `./format.sh`
**Requires**: autopep8

### lint.sh
**Purpose**: Code linting using flake8
**Usage**: `./lint.sh`
**Requires**: flake8

### run_tests.sh
**Purpose**: Run the test suite
**Usage**: `./run_tests.sh`
**Features**: Executes all unit and integration tests

## 🧹 Maintenance Scripts

### cleanup_evolution.sh
**Purpose**: Remove evolution state and cached data
**Usage**: `./cleanup_evolution.sh`
**Features**: Removes entire `./data/evolution/` directory
**Use Case**: Reset evolution system or clean up disk space

### cleanup_logs.sh
**Purpose**: Remove old log files
**Usage**: `./cleanup_logs.sh`
**Features**: Rotates logs older than 30 days

### cleanup_memory.sh
**Purpose**: Clean up memory data and cache
**Usage**: `./cleanup_memory.sh`
**Features**: Safe cleanup of memory files and cache

### cleanup_fresh.sh
**Purpose**: Complete fresh install reset
**Usage**: `./cleanup_fresh.sh`
**Features**: Removes ALL data, logs, and cache - returns to fresh install state
**⚠️ Warning**: Permanently deletes all stored memories and evolution data

## 🧠 Memory Management Scripts

### memory_prune.py
**Purpose**: Remove old/unimportant memories
**Usage**: `python scripts/memory_prune.py`

### memory_consolidate.py
**Purpose**: Merge similar memories
**Usage**: `python scripts/memory_consolidate.py`

### memory_deduplicate.py
**Purpose**: Remove duplicate memories
**Usage**: `python scripts/memory_deduplicate.py`

### memory_forget.py
**Purpose**: Selectively remove memories by criteria
**Usage**: `python scripts/memory_forget.py`

## 📊 Analysis & Business Intelligence Scripts

### business_impact_analyzer.py
**Purpose**: Comprehensive business impact validation and ROI analysis
**Usage**: `python scripts/business_impact_analyzer.py`
**Features**:
- Executive-level business metrics analysis
- Token reduction validation with statistical significance
- Quality improvement measurement and ROI calculation
- Investment recommendations and trend analysis
- Real-time business intelligence dashboards

### performance_analyzer.py
**Purpose**: Analyze system performance and bottlenecks
**Usage**: `python scripts/performance_analyzer.py`
**Documentation**: See `README_performance_analyzer.md`

### iterative_prompts.sh
**Purpose**: Load testing with random questions
**Usage**: `./iterative_prompts.sh`
**Features**: Sends 200 random questions to test memory accumulation

## 🚢 Deployment Scripts

### deploy.sh
**Purpose**: Production deployment automation
**Usage**: `./deploy.sh`
**Features**: Production deployment, environment setup, production configuration

### init_memory_system.py
**Purpose**: Initialize memory system with baseline data
**Usage**: `python scripts/init_memory_system.py`

## 🚀 Evolution & Auto-Optimization

The MemEvolve system now features **intelligent auto-evolution** with multiple triggers:
- **Request-based**: Evolution after configurable request count
- **Performance-based**: Automatic evolution on performance degradation
- **Plateau-based**: Evolution when fitness improvement plateaus
- **Time-based**: Scheduled evolution at regular intervals

Configuration is done via environment variables (see `.env.example`):
- `MEMEVOLVE_AUTO_EVOLUTION_ENABLED`
- `MEMEVOLVE_AUTO_EVOLUTION_REQUESTS`
- `MEMEVOLVE_AUTO_EVOLUTION_DEGRADATION`
- `MEMEVOLVE_AUTO_EVOLUTION_PLATEAU`
- `MEMEVOLVE_AUTO_EVOLUTION_HOURS`

## 📝 Notes

- All scripts assume they're run from the project root directory
- Some scripts require additional dependencies (listed in requirements-dev.txt)
- Maintenance scripts are safe to run on production systems
- Development scripts help maintain code quality
- Business analytics provide executive-level insights for ROI validation
- Auto-evolution continuously optimizes system performance without manual intervention