-- HSOF Database Initialization Script
-- Creates tables for feature selection history and results

-- Feature selection runs table
CREATE TABLE IF NOT EXISTS feature_selection_runs (
    id SERIAL PRIMARY KEY,
    run_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid(),
    dataset_name VARCHAR(255) NOT NULL,
    total_features INTEGER NOT NULL,
    target_features INTEGER NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50) DEFAULT 'running',
    final_score DOUBLE PRECISION,
    gpu_config JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Stage results table
CREATE TABLE IF NOT EXISTS stage_results (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES feature_selection_runs(run_id) ON DELETE CASCADE,
    stage_number INTEGER NOT NULL,
    stage_name VARCHAR(100) NOT NULL,
    input_features INTEGER NOT NULL,
    output_features INTEGER NOT NULL,
    processing_time_seconds DOUBLE PRECISION,
    gpu_utilization_avg DOUBLE PRECISION,
    memory_usage_mb DOUBLE PRECISION,
    selected_features JSONB,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Feature importance history
CREATE TABLE IF NOT EXISTS feature_importance (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES feature_selection_runs(run_id) ON DELETE CASCADE,
    feature_name VARCHAR(255) NOT NULL,
    stage_number INTEGER NOT NULL,
    importance_score DOUBLE PRECISION NOT NULL,
    rank INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- MCTS tree snapshots for analysis
CREATE TABLE IF NOT EXISTS mcts_snapshots (
    id SERIAL PRIMARY KEY,
    run_id UUID REFERENCES feature_selection_runs(run_id) ON DELETE CASCADE,
    iteration INTEGER NOT NULL,
    total_nodes INTEGER NOT NULL,
    best_score DOUBLE PRECISION NOT NULL,
    exploration_rate DOUBLE PRECISION,
    gpu_id INTEGER,
    tree_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Metamodel predictions cache (backup to Redis)
CREATE TABLE IF NOT EXISTS metamodel_cache (
    id SERIAL PRIMARY KEY,
    feature_set_hash VARCHAR(64) UNIQUE NOT NULL,
    predicted_score DOUBLE PRECISION NOT NULL,
    confidence DOUBLE PRECISION,
    model_version VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    accessed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Performance benchmarks
CREATE TABLE IF NOT EXISTS performance_benchmarks (
    id SERIAL PRIMARY KEY,
    benchmark_name VARCHAR(255) NOT NULL,
    dataset_size INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    gpu_model VARCHAR(100),
    stage1_time_seconds DOUBLE PRECISION,
    stage2_time_seconds DOUBLE PRECISION,
    stage3_time_seconds DOUBLE PRECISION,
    total_time_seconds DOUBLE PRECISION,
    peak_memory_usage_gb DOUBLE PRECISION,
    avg_gpu_utilization DOUBLE PRECISION,
    configuration JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_runs_status ON feature_selection_runs(status);
CREATE INDEX idx_runs_dataset ON feature_selection_runs(dataset_name);
CREATE INDEX idx_stage_results_run ON stage_results(run_id);
CREATE INDEX idx_feature_importance_run ON feature_importance(run_id);
CREATE INDEX idx_mcts_snapshots_run ON mcts_snapshots(run_id, iteration);
CREATE INDEX idx_metamodel_cache_accessed ON metamodel_cache(accessed_at);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_feature_selection_runs_updated_at 
    BEFORE UPDATE ON feature_selection_runs 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();