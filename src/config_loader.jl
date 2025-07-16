using YAML

struct HSoFConfig
    dataset_name::String
    database_path::String
    table_name::String
    target_column::String
    id_columns::Vector{String}
    problem_type::String
    stage1_features::Int
    stage2_features::Int
    final_features::Int
end

function load_config(yaml_path::String)::HSoFConfig
    config = YAML.load_file(yaml_path)
    
    return HSoFConfig(
        config["name"],
        config["database"]["path"],
        config["tables"]["train_features"],
        config["target_column"],
        config["id_columns"],
        config["problem_type"],
        get(config, "stage1_features", 50),
        get(config, "stage2_features", 20),
        get(config, "final_features", 10)
    )
end

function validate_config(config::HSoFConfig, n_features::Int)::HSoFConfig
    # Ensure progressive reduction: n_features > stage1 > stage2 > final
    stage1_features = min(config.stage1_features, max(1, n_features - 1))
    stage2_features = min(config.stage2_features, max(1, stage1_features - 1))
    final_features = min(config.final_features, max(1, stage2_features - 1))
    
    return HSoFConfig(
        config.dataset_name,
        config.database_path,
        config.table_name,
        config.target_column,
        config.id_columns,
        config.problem_type,
        stage1_features,
        stage2_features,
        final_features
    )
end