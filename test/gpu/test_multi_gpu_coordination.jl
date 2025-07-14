using Test
using CUDA

# Include the multi-GPU coordination module
include("../../src/gpu/kernels/multi_gpu_coordination.jl")
include("../../src/gpu/kernels/mcts_types.jl")

using .MultiGPUCoordination
using .MCTSTypes

@testset "Multi-GPU Coordination Tests" begin
    
    # Check if we have multiple GPUs
    num_gpus = CUDA.ndevices()
    has_multi_gpu = num_gpus >= 2
    
    if !has_multi_gpu
        @warn "Less than 2 GPUs available, skipping multi-GPU tests"
    end
    
    @testset "GPU Topology Detection" begin
        topology = detect_gpu_topology()
        
        @test topology.num_gpus == num_gpus
        @test size(topology.peer_access) == (num_gpus, num_gpus)
        @test size(topology.peer_bandwidth) == (num_gpus, num_gpus)
        @test size(topology.nvlink_available) == (num_gpus, num_gpus)
        @test length(topology.pcie_gen) == num_gpus
        
        # Diagonal should be self-access
        for i in 1:num_gpus
            @test topology.peer_bandwidth[i, i] > 100.0f0  # Local bandwidth is high
        end
    end
    
    @testset "MultiGPUConfig Creation" begin
        config = MultiGPUConfig(
            Int32(min(2, num_gpus)),
            PARTITION_DEPTH,
            Int32(100),
            Int32(10),
            0.2f0,
            true,
            false,
            Int32(4)
        )
        
        @test config.num_gpus == min(2, num_gpus)
        @test config.partition_strategy == PARTITION_DEPTH
        @test config.sync_interval == 100
        @test config.peer_transfer_threshold == 10
        @test config.load_balance_factor == 0.2f0
        @test config.enable_nvlink == true
        @test config.enable_unified_memory == false
        @test config.max_pending_transfers == 4
    end
    
    @testset "TreePartition Creation" begin
        if num_gpus >= 1
            partition = TreePartition(Int32(1), Int32(2), PARTITION_DEPTH)
            
            @test partition.gpu_id == 1
            @test partition.node_range_start > 0
            @test partition.node_range_end > partition.node_range_start
            @test size(partition.node_owners) == (MCTSTypes.MAX_NODES,)
            @test size(partition.is_local) == (MCTSTypes.MAX_NODES,)
            @test size(partition.remote_children) == (4, MCTSTypes.MAX_NODES)
            @test size(partition.remote_parent) == (MCTSTypes.MAX_NODES,)
        end
    end
    
    @testset "GPU Barrier" begin
        if has_multi_gpu
            barrier = GPUBarrier(Int32(2))
            
            @test barrier.num_gpus == 2
            CUDA.@allowscalar begin
                @test barrier.arrived[1] == 0
                @test barrier.generation[1] == 0
                @test barrier.sense[1] == false
            end
        end
    end
    
    @testset "MultiGPUCoordinator Creation" begin
        if has_multi_gpu
            config = MultiGPUConfig(
                Int32(2),
                PARTITION_SUBTREE,
                Int32(50),
                Int32(5),
                0.15f0,
                false,
                false,
                Int32(2)
            )
            
            coordinator = MultiGPUCoordinator(config)
            
            @test coordinator.config == config
            @test length(coordinator.partitions) == 2
            @test length(coordinator.workload_per_gpu) == 2
            @test length(coordinator.migration_queue) == 2
            
            CUDA.@allowscalar begin
                @test coordinator.sync_iteration[1] == 0
                @test coordinator.total_transfers[1] == 0
                @test coordinator.total_sync_time[1] == 0.0
                @test coordinator.load_imbalance[1] == 0.0f0
            end
        end
    end
    
    @testset "Ownership Update Kernel" begin
        if num_gpus >= 1
            CUDA.device!(0)
            
            node_owners = CUDA.fill(Int32(-1), MCTSTypes.MAX_NODES)
            is_local = CUDA.zeros(Bool, MCTSTypes.MAX_NODES)
            
            gpu_id = Int32(1)
            node_range_start = Int32(1)
            node_range_end = Int32(1000)
            
            @cuda threads=256 blocks=cld(MCTSTypes.MAX_NODES, 256) MultiGPUCoordination.update_ownership_kernel!(
                node_owners,
                is_local,
                gpu_id,
                node_range_start,
                node_range_end
            )
            
            # Check ownership was set correctly
            CUDA.@allowscalar begin
                @test node_owners[1] == 1
                @test node_owners[500] == 1
                @test node_owners[1000] == 1
                @test node_owners[1001] == -1  # Outside range
                
                @test is_local[1] == true
                @test is_local[1000] == true
                @test is_local[1001] == false
            end
        end
    end
    
    @testset "Workload Collection" begin
        if num_gpus >= 1
            CUDA.device!(0)
            
            workload = CUDA.zeros(Int32, 1)
            node_owners = CUDA.zeros(Int32, MAX_NODES)  # Initialize full array
            visit_counts = CUDA.zeros(Int32, MAX_NODES)  # Initialize full array
            
            # Set ownership and visits for first 100 nodes
            CUDA.@allowscalar begin
                for i in 1:100
                    node_owners[i] = 1
                    visit_counts[i] = 5
                end
            end
            
            @cuda threads=256 blocks=cld(MAX_NODES, 256) MultiGPUCoordination.collect_workload_kernel!(
                workload,
                node_owners,
                visit_counts,
                Int32(1)
            )
            
            CUDA.@allowscalar @test workload[1] == 500  # 100 nodes * 5 visits
        end
    end
    
    @testset "Statistics Collection" begin
        if has_multi_gpu
            config = MultiGPUConfig(
                Int32(2),
                PARTITION_WORKLOAD,
                Int32(100),
                Int32(10),
                0.2f0,
                false,
                false,
                Int32(4)
            )
            
            coordinator = MultiGPUCoordinator(config)
            
            # Simulate some activity
            CUDA.@allowscalar begin
                coordinator.total_transfers[1] = 1000
                coordinator.total_sync_time[1] = 1.5
                coordinator.load_imbalance[1] = 0.15f0
                coordinator.sync_iteration[1] = 50
                
                coordinator.workload_per_gpu[1][1] = 5000
                coordinator.workload_per_gpu[2][1] = 4500
            end
            
            stats = get_multi_gpu_stats(coordinator)
            
            @test stats["num_gpus"] == 2
            @test stats["total_transfers"] == 1000
            @test stats["total_sync_time"] == 1.5
            @test stats["load_imbalance"] == 0.15f0
            @test stats["sync_iteration"] == 50
            @test stats["workload_per_gpu"] == [5000, 4500]
            @test haskey(stats, "nvlink_connections")
            @test haskey(stats, "peer_access_pairs")
        end
    end
    
    @testset "Single GPU Fallback" begin
        if num_gpus == 1
            # Test that single GPU configuration works
            config = MultiGPUConfig(
                Int32(1),
                PARTITION_DEPTH,
                Int32(100),
                Int32(10),
                0.2f0,
                false,
                false,
                Int32(4)
            )
            
            coordinator = MultiGPUCoordinator(config)
            
            @test coordinator.config.num_gpus == 1
            @test length(coordinator.partitions) == 1
            
            # Single GPU should own all nodes
            partition = coordinator.partitions[1]
            @test partition.node_range_start == 1
            @test partition.node_range_end == MCTSTypes.MAX_NODES
        end
    end
end

println("\n✅ Multi-GPU coordination tests completed!")