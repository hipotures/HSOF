# Hybrid Search for Optimal Features (HSOF) Product Requirements Document

## 1. Introduction

This Product Requirements Document (PRD) outlines the comprehensive requirements for HSOF (Hybrid Search for Optimal Features), an advanced multi-stage feature selection tool that combines GPU-accelerated Monte Carlo Tree Search with metamodel evaluation and progressive feature reduction strategies. This document serves as the primary reference for development teams, ML engineers, and stakeholders to ensure successful delivery of a revolutionary hybrid feature selection platform.

The document covers all functional and technical requirements, user stories, acceptance criteria, and implementation details necessary to build a high-performance feature selection system that intelligently combines multiple search strategies to handle datasets with 500-5000+ features.

## 2. Product overview

HSOF (Hybrid Search for Optimal Features) is a GPU-accelerated multi-stage feature selection platform designed for data scientists and ML engineers working with high-dimensional datasets. The system employs a hybrid approach combining:

1. **Stage 1**: Fast univariate filtering (5000→500 features)
2. **Stage 2**: GPU-MCTS with metamodel evaluation (500→50 features)  
3. **Stage 3**: Precise evaluation with real models (50→10-20 features)

The platform leverages persistent CUDA kernels, metamodel-based evaluation, and ensemble tree search to intelligently navigate the feature space. Users can analyze datasets, discover optimal feature combinations through progressive reduction, and visualize the multi-stage selection process through an intuitive Rich console dashboard.

The tool adapts its strategy based on dataset characteristics, using appropriate methods at each reduction stage to balance speed and accuracy. Premium users can access advanced features including custom metamodel architectures, distributed multi-GPU processing, and automated hyperparameter optimization for each stage.

### Key value propositions:
- Hybrid multi-stage approach optimized for different scales
- 100-1000x speedup through GPU acceleration and smart reduction
- Metamodel-based evaluation eliminating repeated model training
- Real-time console dashboard with Rich UI
- Scalable to datasets with 5000+ features through staged reduction
- Direct SQLite database integration with existing feature stores

## 3. Goals and objectives

### Primary goals:
- Enable efficient feature selection through intelligent multi-stage reduction
- Achieve >80% sustained GPU utilization during MCTS exploration phase
- Provide accurate feature selection for datasets with 500-5000+ features
- Deliver intuitive Rich console dashboard for real-time monitoring
- Seamlessly integrate with existing SQLite feature databases

### Success metrics:
- Stage 1 reduction: 5000→500 features in <30 seconds
- Stage 2 GPU utilization: >80% sustained during MCTS
- Stage 3 accuracy: Equal or better than exhaustive search
- Overall processing: <2 minutes for 5000 features → optimal subset
- Memory efficiency: <8GB VRAM per GPU for datasets up to 5000 features
- Database overhead: <1% performance impact from SQLite operations

### Technical objectives:
- Zero CPU-GPU transfer during MCTS exploration
- Efficient stage transitions with minimal data movement
- Metamodel accuracy >0.9 correlation with true model scores
- Support for dual RTX 4090 setup without NVLink
- Robust checkpoint system for long-running searches

## 4. Target audience

### Primary users:
- **Data scientists**: Professionals working with high-dimensional datasets requiring efficient hybrid feature selection
- **ML engineers**: Engineers optimizing model pipelines through multi-stage feature reduction
- **Researchers**: Academic researchers exploring feature interactions in complex datasets
- **Kaggle competitors**: Data science competition participants needing fast, accurate feature engineering

### Secondary users:
- **AutoML platforms**: Companies integrating automated feature selection into ML pipelines
- **Financial institutions**: Quants selecting features for trading models
- **Biotech companies**: Researchers analyzing genomic data with thousands of features
- **Tech companies**: Teams optimizing recommendation systems and search algorithms

### User characteristics:
- Technical proficiency: Advanced (comfortable with Python/Julia, basic CUDA knowledge)
- Dataset sizes: 1K-1M samples, 500-5000 features
- Hardware access: NVIDIA GPU with 8GB+ VRAM
- Primary pain points: Slow feature selection, inability to explore feature interactions, computational bottlenecks

## 5. Features and requirements

### 5.1 Core features

#### Hybrid multi-stage selection pipeline
- **Stage 1**: Fast filtering (5000→500 features)
  - Mutual information calculation on GPU
  - Univariate statistical tests  
  - Correlation-based filtering
  - Variance thresholding
- **Stage 2**: GPU-MCTS exploration (500→50 features)
  - Persistent CUDA kernels
  - Metamodel-based evaluation
  - Ensemble tree search
- **Stage 3**: Precise evaluation (50→final features)
  - Real model training (XGBoost/RF)
  - Full cross-validation
  - Feature interaction analysis

#### GPU-native MCTS engine
- Persistent CUDA kernels for continuous GPU execution
- Structure of Arrays (SoA) memory layout for coalesced access
- Lock-free parallel tree operations
- Warp-level primitives for efficient reductions

#### Metamodel evaluation system
- **Purpose**: Predict cross-validation scores without training actual models (XGBoost/RF)
- **Architecture**: Neural network that learns to approximate model performance
  ```
  Input: Binary vector [0,1,0,1,1,0,...] indicating selected features (500 dims)
  → Dense(256) + ReLU + Dropout(0.2)
  → Multi-Head Attention (8 heads) - captures feature interactions
  → Dense(128) + ReLU + Dropout(0.2)  
  → Dense(64) + ReLU
  → Dense(1) + Sigmoid
  Output: Predicted CV score [0.0-1.0]
  ```
- **Training Strategy**:
  1. Pre-training: 10,000 random feature combinations with real scores
  2. Online learning: Update every 100 MCTS iterations
  3. Experience replay buffer: Last 1000 evaluations
- **Performance**: 1000x faster than actual model training (0.001s vs 1s)
- **Accuracy Target**: >0.9 correlation with true model scores
- **GPU Optimization**: Batch evaluation of 1000+ combinations in parallel

#### SQLite database integration
- Direct connection to existing feature database
- Lazy loading with configurable chunk size
- Metadata-driven configuration
- Minimal write frequency to avoid GPU stalls
- Write strategies:
  - Results: Write once at completion
  - Checkpoints: Every 5 minutes or 10K iterations
  - Progress: In-memory only, no DB writes
  - Logs: Buffered writes every 1000 entries

#### Ensemble forest management
- Parallel execution of 100+ MCTS trees
- Diversity mechanisms for exploration coverage
- Consensus building across trees
- Dynamic load balancing

#### User interface
- Web-based dashboard with real-time updates
- Dataset upload and preprocessing
- Feature selection configuration
- Progress monitoring and control

#### Visualization and analytics
- Interactive tree visualization with D3.js
- Feature importance heatmaps
- Convergence plots and metrics
- Export functionality for results

### 5.2 Premium features

#### Advanced metamodel architectures
- Transformer-based feature interaction modeling
- Graph neural networks for feature relationships
- Custom architecture design interface
- Pre-trained model marketplace

#### Multi-GPU scaling
- Distributed tree forest across multiple GPUs
- Automatic workload distribution
- Linear scaling efficiency >85%
- Support for mixed GPU configurations

#### Hyperparameter optimization
- Automated tuning for each stage
- Bayesian optimization for stage transitions
- Grid search parallelization on GPU
- Performance profiling per stage

### 5.3 Hybrid approach benefits

#### Computational efficiency
- Stage 1: O(n×p) complexity for p features, parallelized on GPU
- Stage 2: MCTS explores only promising subspace (10% of original)
- Stage 3: Precise evaluation on <100 candidates only
- Overall: 100-1000x faster than exhaustive search

#### Accuracy advantages
- No single method bias - combines strengths of multiple approaches
- Coarse-to-fine strategy prevents missing important features
- Metamodel guides search while real models validate
- Progressive reduction maintains interpretability

#### Scalability
- Handles 5000+ features through intelligent staging
- Memory-efficient transitions between stages
- Adaptive strategy based on dataset characteristics
- Graceful degradation for extremely large feature spaces

## 6. User stories and acceptance criteria

### 6.1 Core functionality user stories

**HSOF-101: Dataset upload and preprocessing**
- **User story**: As a data scientist, I want to upload my dataset so that I can perform hybrid feature selection
- **Acceptance criteria**:
  - Support for CSV, Parquet, and SQLite database connections
  - Automatic feature type detection (numeric, categorical)
  - Missing value handling with multiple strategies
  - Data validation with error reporting
  - GPU-accelerated preprocessing pipeline

**HSOF-102: Multi-stage configuration**
- **User story**: As a user, I want to configure each stage of the hybrid search so that I can optimize for my dataset
- **Acceptance criteria**:
  - Stage 1: Configure filtering thresholds (MI, correlation, variance)
  - Stage 2: MCTS parameters (trees, exploration, metamodel)
  - Stage 3: Model selection (XGBoost, RF, LightGBM)
  - Automatic stage transition criteria
  - Manual override options for stage boundaries

**HSOF-103: Real-time hybrid search execution**
- **User story**: As a user, I want to see real-time progress across all stages so that I can monitor the selection process
- **Acceptance criteria**:
  - Live stage indicator showing current phase
  - Features remaining at each stage
  - Best feature combinations per stage
  - GPU utilization for Stage 2
  - Estimated time to completion

**HSOF-104: Result visualization**
- **User story**: As a user, I want to visualize the hybrid search results so that I can understand the selection process
- **Acceptance criteria**:
  - Stage-by-stage feature reduction funnel
  - Interactive tree visualization for MCTS stage
  - Feature importance across all stages
  - Comparison of stage selections
  - Export to PNG/SVG formats

**HSOF-105: Feature importance analysis**
- **User story**: As a user, I want detailed feature importance metrics so that I can interpret results
- **Acceptance criteria**:
  - Individual feature importance scores
  - Pairwise interaction strengths
  - Feature clustering visualization
  - Statistical significance testing
  - Comparison with baseline methods

**FS-106: SQLite database integration**
- **User story**: As a user, I want to connect to my existing SQLite database so that I can use pre-extracted features
- **Acceptance criteria**:
  - Database connection configuration UI
  - Table selection from available tables
  - Automatic metadata reading (excluded columns, ID, target)
  - Column validation and type checking
  - Streaming data loader for large tables (>1GB)
  - Progress indicator for data loading
  - Error handling for missing/corrupt data

### 6.2 GPU computation user stories

**FS-201: Persistent kernel execution**
- **User story**: As a system, I need persistent GPU kernels so that I can minimize CPU-GPU communication
- **Acceptance criteria**:
  - Kernel execution without CPU intervention for 1000+ iterations
  - Atomic operations for thread synchronization
  - Shared memory optimization for tree nodes
  - Warp-level reduction operations
  - Dynamic parallelism for adaptive exploration

**FS-202: Metamodel inference**
- **User story**: As a system, I need fast metamodel evaluation so that I can assess feature combinations quickly
- **Acceptance criteria**:
  - Batch inference for 1000+ combinations simultaneously
  - Mixed precision (FP16) computation support
  - Tensor Core utilization on compatible GPUs
  - <1ms latency per batch evaluation
  - Online weight updates without kernel restart

**FS-203: Memory management**
- **User story**: As a system, I need efficient GPU memory usage so that I can handle large datasets
- **Acceptance criteria**:
  - Memory pooling for dynamic allocation
  - Automatic garbage collection for expired nodes
  - Virtual memory support for large trees
  - Memory bandwidth optimization >80% efficiency
  - Out-of-memory handling with graceful degradation

**FS-204: Multi-stream coordination**
- **User story**: As a system, I need multiple CUDA streams so that I can parallelize tree operations
- **Acceptance criteria**:
  - Independent streams for each MCTS tree
  - Asynchronous kernel execution
  - Stream synchronization for result aggregation
  - Load balancing across streams
  - Priority scheduling for promising trees

### 6.3 Premium feature user stories

**FS-301: Multi-GPU scaling**
- **User story**: As a premium user, I want to use multiple GPUs so that I can accelerate large-scale selection
- **Acceptance criteria**:
  - Automatic GPU detection and initialization
  - Work distribution across 2-8 GPUs
  - Inter-GPU communication optimization
  - Linear scaling efficiency >85%
  - Fault tolerance for GPU failures

**FS-302: Custom metamodel design**
- **User story**: As a premium user, I want to design custom metamodels so that I can optimize for my domain
- **Acceptance criteria**:
  - Visual neural architecture builder
  - Pre-built architecture templates
  - Custom loss function definition
  - Training progress monitoring
  - Model versioning and comparison

**FS-303: Advanced analytics**
- **User story**: As a premium user, I want advanced analytics so that I can gain deeper insights
- **Acceptance criteria**:
  - Feature interaction 3D visualization
  - Causal relationship inference
  - Temporal stability analysis
  - Cross-validation performance tracking
  - Automated report generation

**FS-304: API access**
- **User story**: As a premium user, I want API access so that I can integrate with my ML pipeline
- **Acceptance criteria**:
  - RESTful API for job submission
  - WebSocket for real-time updates
  - Batch processing endpoints
  - Result webhook notifications
  - Rate limiting and authentication

### 6.4 Performance and reliability user stories

**FS-401: High availability**
- **User story**: As a user, I want reliable service so that my feature selection jobs complete successfully
- **Acceptance criteria**:
  - 99.9% uptime SLA
  - Automatic job recovery after failures
  - Checkpoint saving every 100 iterations
  - Result persistence for 30 days
  - Email notifications for job completion

**FS-402: Performance optimization**
- **User story**: As a user, I want optimal performance so that I can get results quickly
- **Acceptance criteria**:
  - Automatic performance profiling
  - Hardware-specific optimization
  - Kernel auto-tuning
  - Memory layout optimization
  - Bandwidth utilization reporting

## 7. Technical requirements / Stack

### 7.1 Frontend technology stack

#### Console Dashboard (Primary Interface)
- **Framework**: Rich (Python) for terminal UI
- **Layout**: Multi-panel dashboard with real-time updates
- **Components**: Tables, progress bars, sparklines, gauges
- **Update Rate**: 100ms refresh for smooth animations
- **Color Support**: 256-color and true color terminals

#### Dashboard Components:
1. **GPU Status Panel** (Top, 2x side-by-side for dual GPU)
   - GPU utilization gauge (0-100%)
   - Memory usage bar (used/total GB)
   - Temperature indicator with color coding
   - Power draw (W) and clock speeds
   - Current kernel status
   - SM occupancy percentage
   - Memory bandwidth utilization

2. **MCTS Progress Panel** (Center-left)
   - Total iterations counter
   - Best score tracker with sparkline graph
   - Top 5 feature combinations table with scores
   - Convergence indicator (iterations since improvement)
   - Trees explored counter per GPU
   - Unique feature combinations evaluated
   - Metamodel accuracy tracking

3. **Performance Metrics Panel** (Center-right)
   - Nodes evaluated/second (with trend)
   - Cache hit rate percentage
   - Memory bandwidth utilization
   - PCIe transfer rate between GPUs
   - Iteration time histogram
   - Metamodel inference time
   - Database read throughput (MB/s)

4. **Feature Analysis Panel** (Bottom-left)
   - Most visited features horizontal bar chart
   - Feature correlation mini-heatmap
   - Current exploration depth
   - Feature count in best combination
   - Feature interaction strength matrix
   - Excluded features counter

5. **System Log Panel** (Bottom-right)
   - Rolling log with timestamps
   - Error/warning highlighting with colors
   - Checkpoint save notifications
   - Database sync status
   - GPU kernel launch events
   - Memory allocation warnings
   - Temperature throttling alerts

#### Web Interface (Secondary)
- **Framework**: Next.js 15 with TypeScript
- **Visualization**: D3.js for tree visualization
- **Purpose**: Detailed analysis and result export

### 7.2 Backend technology stack
- **Runtime**: Node.js with TypeScript for API server
- **GPU Runtime**: Julia 1.10+ with CUDA.jl
- **API Framework**: Express.js with OpenAPI documentation
- **Job Queue**: Bull with Redis backend
- **Database**: PostgreSQL for job metadata
- **Object Storage**: S3-compatible for datasets

### 7.3 Data storage and persistence

#### SQLite Database Schema
- **Primary Database**: Existing product database with feature extraction
  ```sql
  -- Main dataset table (user's existing table)
  CREATE TABLE dataset_features (
    id INTEGER PRIMARY KEY,
    [feature columns...],  -- 500-5000 feature columns
    [target column],       -- As specified in metadata
    created_at TIMESTAMP
  );
  
  -- Metadata table (user's existing table)
  CREATE TABLE dataset_metadata (
    table_name TEXT PRIMARY KEY,
    excluded_columns TEXT,  -- JSON array of excluded column names
    id_columns TEXT,        -- JSON array of ID column names  
    target_column TEXT,     -- Name of target column
    feature_count INTEGER,
    row_count INTEGER,
    last_updated TIMESTAMP
  );
  
  -- Example metadata entry:
  -- {
  --   "table_name": "titanic_features_v3",
  --   "excluded_columns": ["PassengerId", "Name", "Ticket"],
  --   "id_columns": ["PassengerId"],
  --   "target_column": "Survived",
  --   "feature_count": 127,
  --   "row_count": 891
  -- }
  
  -- MCTS results table (new)
  CREATE TABLE mcts_results (
    run_id TEXT PRIMARY KEY,
    dataset_table TEXT,
    best_features TEXT,     -- JSON array of column names
    best_score REAL,
    baseline_score REAL,    -- Score with all features
    improvement REAL,       -- Percentage improvement
    total_iterations INTEGER,
    convergence_time REAL,
    gpu_config TEXT,        -- JSON with GPU setup
    parameters TEXT,        -- JSON with MCTS parameters
    created_at TIMESTAMP
  );
  
  -- Checkpoint table (sparse writes)
  CREATE TABLE mcts_checkpoints (
    checkpoint_id INTEGER PRIMARY KEY,
    run_id TEXT,
    iteration INTEGER,
    tree_state BLOB,        -- Compressed tree state
    best_features TEXT,     -- Current best feature set
    best_score REAL,
    gpu0_throughput REAL,   -- Nodes/sec GPU 0
    gpu1_throughput REAL,   -- Nodes/sec GPU 1
    timestamp TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES mcts_results(run_id)
  );
  
  -- Feature importance table (written at completion)
  CREATE TABLE feature_importance (
    run_id TEXT,
    feature_name TEXT,
    importance_score REAL,
    visit_count INTEGER,
    avg_score_with REAL,
    avg_score_without REAL,
    PRIMARY KEY (run_id, feature_name),
    FOREIGN KEY (run_id) REFERENCES mcts_results(run_id)
  );
  ```

#### Database Integration Requirements
- **Connection Pooling**: Read-only pool for dataset access
- **Lazy Loading**: Stream large datasets in chunks
- **Write Strategy**: Batch writes every 1000 iterations
- **Checkpoint Frequency**: Every 5 minutes or 10K iterations
- **Transaction Management**: ACID compliance for checkpoints

#### GPU computation stack
- **CUDA**: Version 11.8+ with compute capability 7.0+
- **Libraries**: cuBLAS, cuDNN, NCCL for multi-GPU
- **Julia Packages**: CUDA.jl, Flux.jl, MLJ.jl
- **Memory Management**: CUDA memory pools
- **Profiling**: NSight Systems and Compute

### 7.4 Infrastructure requirements
- **GPU Hardware**: 2x RTX 4090 (24GB each, no NVLink)
- **Multi-GPU Communication**: PCIe 4.0 x16 for both cards
- **GPU Strategy**: Independent tree forests per GPU
- **CPU**: 16+ cores for data preprocessing
- **RAM**: 64GB+ for large datasets
- **Storage**: NVMe SSD for fast SQLite I/O

#### Multi-GPU Architecture (2x RTX 4090)
- **Distribution Strategy**: 
  - GPU 0: Trees 1-50 + Metamodel training
  - GPU 1: Trees 51-100 + Metamodel inference
- **Communication**: Minimal PCIe transfers (top candidates only)
- **Synchronization**: Every 1000 iterations via CPU
- **Memory Layout**: Duplicate dataset on both GPUs
- **Fault Tolerance**: Continue on single GPU if one fails

### 7.5 Security requirements
- **Authentication**: JWT with refresh tokens
- **Authorization**: Role-based access control
- **Data Encryption**: AES-256 for datasets at rest
- **API Security**: Rate limiting, input validation
- **Compliance**: SOC 2 Type II certification

### 7.6 Deployment architecture
- **Container**: Docker with NVIDIA runtime
- **Orchestration**: Kubernetes with GPU support
- **CI/CD**: GitLab CI with GPU runners
- **Monitoring**: Prometheus + Grafana
- **Logging**: ELK stack with GPU metrics

## 8. Design and user interface

### 8.1 Design principles
- **Performance-first**: Real-time updates without UI lag
- **Data-dense**: Maximum information density for power users
- **Accessibility**: Keyboard shortcuts for all actions
- **Customizable**: User-configurable layouts and themes
- **Responsive**: Adaptive to screen size and GPU capabilities

### 8.2 Console Dashboard Interface

#### Rich Terminal UI Layout
```
╔══════════════════════════════════════════════════════════════════════════════╗
║  HSOF - Hybrid Search for Optimal Features v1.0  |  Stage 2/3: GPU-MCTS      ║
╠══════════════════════════════════════╦══════════════════════════════════════╣
║         GPU 0 Status                 ║         GPU 1 Status                 ║
║ Util: ████████░░ 85%  Mem: 18.5/24GB║ Util: ███████░░░ 78%  Mem: 17.2/24GB║
║ Temp: 72°C  Power: 320W  Clock: 2.5GHz║ Temp: 69°C  Power: 305W  Clock: 2.5GHz║
╠══════════════════════════════════════╩══════════════════════════════════════╣
║                     Hybrid Search Progress                                   ║
║ Stage 1: ✓ 5000→487 features (12.3s)  |  Stage 2: ● 487→52 (45s/~2m)       ║
║ Stage 3: ○ Pending                     |  Best Score: 0.9234 ▆▇█▇▆▅▆▇█     ║
║ ┌─────────────────────────────────────────────────────────────────────┐   ║
║ │ Top Feature Combinations                Score   Visits   Stage      │   ║
║ │ 1. [Age, Sex, Fare, Pclass]           0.9234   1,245      2        │   ║
║ │ 2. [Age, Sex, Cabin, Embarked]        0.9187   1,102      2        │   ║
║ │ 3. [Fare, Pclass, Title, Family]      0.9156     987      2        │   ║
║ └─────────────────────────────────────────────────────────────────────┘   ║
╠═══════════════════════════════════════╦═════════════════════════════════════╣
║      Performance Metrics              ║         Feature Analysis            ║
║ Nodes/sec: 1,234,567                 ║ Features by Stage:                  ║
║ Cache Hit: 89.4%                     ║ Stage 1: 5000 → 487 (90.3% reduced) ║
║ PCIe: ▂▄▆█▇▅▃ 8.2 GB/s              ║ Stage 2: 487 → 52 (89.3% reduced)  ║
║ Bandwidth: 487.3 GB/s                ║ Stage 3: Pending                    ║
╠═══════════════════════════════════════╩═════════════════════════════════════╣
║                              System Log                                      ║
║ [14:23:45] INFO: Stage 2 checkpoint saved (iteration 45,000)               ║
║ [14:23:42] INFO: New best score found: 0.9234                              ║
║ [14:23:38] WARN: GPU 1 memory usage high (>90%)                            ║
║ [14:23:30] INFO: Stage 1 completed: 5000→487 features in 12.3s             ║
╚═════════════════════════════════════════════════════════════════════════════╝
[Q]uit [P]ause [S]ave [E]xport [C]onfig [H]elp [1-3]Stage View
```

#### Dashboard Controls
- **Keyboard Shortcuts**:
  - `Q`: Quit and save final results
  - `P`: Pause/Resume GPU computation
  - `S`: Force checkpoint save
  - `E`: Export current results to CSV
  - `C`: Open configuration menu
  - `H`: Show help overlay
  - `1/2`: Focus on GPU 0/1 details
  - `↑↓`: Scroll through logs
  - `Tab`: Cycle through panels

#### Update Strategies
- **Refresh Rate**: 10 FPS (100ms) for smooth updates
- **Data Aggregation**: 1-second windows for metrics
- **Log Buffer**: Last 100 entries with rotation
- **Sparklines**: 60-second rolling window
- **Color Coding**: Green (good), Yellow (warning), Red (critical)

### 8.3 Layout and navigation
- **Tree visualization**: Force-directed graph with WebGL rendering
- **Heatmaps**: Feature correlation and importance matrices
- **Time series**: Convergence plots with confidence intervals
- **3D scatter**: Feature embedding visualization
- **Network graphs**: Feature interaction networks

### 8.4 Color scheme and theming
- **Dark mode default**: Reduced eye strain for long sessions
- **Semantic colors**: Green (good), yellow (warning), red (critical)
- **GPU status colors**: Utilization-based gradient
- **Accessibility**: High contrast mode available
- **Custom themes**: User-definable color schemes

### 8.5 Interactive elements
- **Drag-and-drop**: Dataset upload, node rearrangement
- **Context menus**: Right-click actions on visualizations
- **Tooltips**: Detailed information on hover
- **Keyboard shortcuts**: Vim-style navigation
- **Gesture support**: Pinch-to-zoom, swipe navigation

### 8.6 Performance indicators
- **GPU utilization gauge**: Real-time percentage display
- **Memory usage bar**: VRAM allocation visualization
- **Throughput counter**: Nodes evaluated per second
- **Progress rings**: Iteration and convergence progress
- **Heat indicators**: Temperature monitoring

### 8.7 Responsive behavior
- **Desktop**: Full feature set with multi-panel layout
- **Tablet**: Simplified layout with collapsible panels
- **Mobile**: Read-only results view with basic controls
- **4K support**: High-DPI optimized visualizations
- **Multi-monitor**: Detachable panels for extended desktop