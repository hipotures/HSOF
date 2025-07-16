"""
Test Suite for Convergence Detection Algorithm
Validates convergence metrics tracking, sliding window analysis, early stopping criteria,
adaptive convergence thresholds, and robust ensemble termination for MCTS ensemble.
"""

using Test
using Random
using Statistics
using Dates

# Include the convergence detection module
include("../../src/stage2/convergence_detection.jl")
using .ConvergenceDetection

# Mock consensus manager for testing
include("../../src/stage2/consensus_voting.jl")
using .ConsensusVoting

@testset "Convergence Detection Algorithm Tests" begin
    
    Random.seed!(42)  # For reproducible tests
    
    @testset "Configuration Tests" begin
        # Test default configuration
        config = create_convergence_config()
        
        @test config.convergence_strategy == HYBRID_CONVERGENCE
        @test config.stopping_criteria == ADAPTIVE_STOPPING
        @test config.threshold_adaptation == PROBLEM_ADAPTIVE
        @test config.stability_threshold == 0.95f0
        @test config.consensus_threshold == 0.9f0
        @test config.entropy_threshold == 0.1f0
        @test config.agreement_threshold == 0.85f0
        @test config.sliding_window_size == 100
        @test config.minimum_iterations == 1000
        @test config.maximum_iterations == 50000
        @test config.early_stopping_enabled == true
        @test config.minimum_stable_windows == 5
        
        # Test custom configuration
        custom_config = create_convergence_config(
            convergence_strategy = STABILITY_BASED,
            stopping_criteria = FIXED_ITERATIONS,
            threshold_adaptation = STATIC_THRESHOLD,
            stability_threshold = 0.9f0,
            sliding_window_size = 50
        )
        
        @test custom_config.convergence_strategy == STABILITY_BASED
        @test custom_config.stopping_criteria == FIXED_ITERATIONS
        @test custom_config.threshold_adaptation == STATIC_THRESHOLD
        @test custom_config.stability_threshold == 0.9f0
        @test custom_config.sliding_window_size == 50
        
        println("  ✅ Configuration tests passed")
    end
    
    @testset "Enum Value Tests" begin
        # Test convergence strategy enum values
        @test Int(STABILITY_BASED) == 1
        @test Int(CONSENSUS_BASED) == 2
        @test Int(ENTROPY_BASED) == 3
        @test Int(HYBRID_CONVERGENCE) == 4
        @test Int(ADAPTIVE_CONVERGENCE) == 5
        
        # Test stopping criteria enum values
        @test Int(FIXED_ITERATIONS) == 1
        @test Int(STATISTICAL_TEST) == 2
        @test Int(PLATEAU_DETECTION) == 3
        @test Int(ENSEMBLE_AGREEMENT) == 4
        @test Int(ADAPTIVE_STOPPING) == 5
        
        # Test threshold adaptation enum values
        @test Int(STATIC_THRESHOLD) == 1
        @test Int(LINEAR_ADAPTATION) == 2
        @test Int(EXPONENTIAL_DECAY) == 3
        @test Int(PROBLEM_ADAPTIVE) == 4
        @test Int(PERFORMANCE_BASED) == 5
        
        println("  ✅ Enum value tests passed")
    end
    
    @testset "Convergence Metrics Creation Tests" begin
        metrics = ConvergenceDetection.create_convergence_metrics()
        
        @test metrics.feature_stability == 0.0
        @test metrics.consensus_strength == 0.0
        @test metrics.selection_entropy == 1.0
        @test metrics.iteration_variance == 1.0
        @test metrics.agreement_coefficient == 0.0
        @test metrics.trend_slope == 0.0
        @test metrics.confidence_interval == (0.0, 1.0)
        @test metrics.update_count == 0
        @test metrics.is_converged == false
        
        println("  ✅ Convergence metrics creation tests passed")
    end
    
    @testset "Sliding Window Creation Tests" begin
        window = ConvergenceDetection.create_convergence_window(30)
        
        @test window.window_size == 30
        @test isempty(window.stability_history)
        @test isempty(window.consensus_history)
        @test isempty(window.entropy_history)
        @test isempty(window.timestamps)
        @test window.window_variance == 0.0
        @test window.window_trend == 0.0
        @test window.is_stable == false
        @test window.consecutive_stable == 0
        
        println("  ✅ Sliding window creation tests passed")
    end
    
    @testset "Detector Initialization Tests" begin
        config = create_convergence_config(sliding_window_size = 20)
        detector = initialize_convergence_detector(config)
        
        @test detector.config == config
        @test detector.is_converged == false
        @test detector.convergence_confidence == 0.0
        @test isnothing(detector.convergence_timestamp)
        @test detector.consecutive_stable_iterations == 0
        @test detector.detector_state == "active"
        @test haskey(detector.current_thresholds, "stability")
        @test haskey(detector.current_thresholds, "consensus")
        @test haskey(detector.current_thresholds, "entropy")
        @test haskey(detector.current_thresholds, "agreement")
        @test detector.sliding_window.window_size == 20
        @test detector.stats.total_iterations == 0
        
        println("  ✅ Detector initialization tests passed")
    end
    
    @testset "Feature Stability Calculation Tests" begin
        config = create_convergence_config()
        detector = initialize_convergence_detector(config)
        
        # Test with identical feature selections (high stability)
        identical_selections = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5],
            [1, 2, 3, 4, 5]
        ]
        stability = ConvergenceDetection.calculate_feature_stability(detector, identical_selections)
        @test stability == 1.0
        
        # Test with completely different selections (low stability)
        different_selections = [
            [1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10],
            [11, 12, 13, 14, 15]
        ]
        stability = ConvergenceDetection.calculate_feature_stability(detector, different_selections)
        @test stability == 0.0
        
        # Test with partially overlapping selections
        partial_selections = [
            [1, 2, 3, 4, 5],
            [1, 2, 3, 6, 7],
            [1, 2, 8, 9, 10]
        ]
        stability = ConvergenceDetection.calculate_feature_stability(detector, partial_selections)
        @test 0.0 < stability < 1.0
        
        # Test with empty selections
        empty_selections = Vector{Vector{Int}}()
        stability = ConvergenceDetection.calculate_feature_stability(detector, empty_selections)
        @test stability == 0.0
        
        # Test caching
        cached_stability = ConvergenceDetection.calculate_feature_stability(detector, identical_selections)
        @test cached_stability == 1.0
        
        println("  ✅ Feature stability calculation tests passed")
    end
    
    @testset "Selection Entropy Calculation Tests" begin
        config = create_convergence_config()
        detector = initialize_convergence_detector(config)
        
        # Test with uniform selections (high entropy)
        uniform_selections = [
            [1], [2], [3], [4], [5]
        ]
        entropy = ConvergenceDetection.calculate_selection_entropy(detector, uniform_selections)
        @test entropy > 0.8  # Should be high entropy
        
        # Test with concentrated selections (low entropy)
        concentrated_selections = [
            [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]
        ]
        entropy = ConvergenceDetection.calculate_selection_entropy(detector, concentrated_selections)
        @test entropy == 0.0  # Only one feature selected
        
        # Test with mixed selections
        mixed_selections = [
            [1, 2], [1, 3], [1, 4], [2, 3], [2, 4]
        ]
        entropy = ConvergenceDetection.calculate_selection_entropy(detector, mixed_selections)
        @test 0.0 < entropy < 1.0
        
        # Test with empty selections
        empty_selections = Vector{Vector{Int}}()
        entropy = ConvergenceDetection.calculate_selection_entropy(detector, empty_selections)
        @test entropy == 1.0  # Maximum entropy for empty
        
        println("  ✅ Selection entropy calculation tests passed")
    end
    
    @testset "Agreement Coefficient Calculation Tests" begin
        config = create_convergence_config()
        detector = initialize_convergence_detector(config)
        
        # Test with equal-sized selections (high agreement)
        equal_size_selections = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
        agreement = ConvergenceDetection.calculate_agreement_coefficient(detector, equal_size_selections)
        @test agreement == 1.0
        
        # Test with varying-sized selections (lower agreement)
        varying_size_selections = [
            [1, 2],
            [3, 4, 5, 6],
            [7]
        ]
        agreement = ConvergenceDetection.calculate_agreement_coefficient(detector, varying_size_selections)
        @test 0.0 < agreement < 1.0
        
        # Test with single selection
        single_selection = [[1, 2, 3]]
        agreement = ConvergenceDetection.calculate_agreement_coefficient(detector, single_selection)
        @test agreement == 1.0
        
        println("  ✅ Agreement coefficient calculation tests passed")
    end
    
    @testset "Convergence Metrics Update Tests" begin
        config = create_convergence_config()
        detector = initialize_convergence_detector(config)
        
        # Create mock consensus manager
        consensus_config = ConsensusVoting.create_consensus_config()
        consensus_manager = ConsensusVoting.initialize_consensus_manager(consensus_config)
        
        # Add some votes to consensus manager
        ConsensusVoting.cast_vote!(consensus_manager, 1, 10, 0.8, 5)
        ConsensusVoting.cast_vote!(consensus_manager, 2, 10, 0.7, 6)
        ConsensusVoting.build_consensus!(consensus_manager)
        
        # Test feature selections
        feature_selections = [
            [10, 20, 30],
            [10, 20, 31],
            [10, 21, 30]
        ]
        
        initial_update_count = detector.metrics.update_count
        ConvergenceDetection.update_convergence_metrics!(detector, feature_selections, consensus_manager)
        
        @test detector.metrics.update_count == initial_update_count + 1
        @test detector.metrics.feature_stability > 0.0
        @test detector.metrics.consensus_strength >= 0.0
        @test detector.metrics.selection_entropy >= 0.0
        @test detector.metrics.agreement_coefficient > 0.0
        @test detector.stats.total_iterations == 1
        
        println("  ✅ Convergence metrics update tests passed")
    end
    
    @testset "Sliding Window Update Tests" begin
        config = create_convergence_config(sliding_window_size = 5)
        detector = initialize_convergence_detector(config)
        
        # Manually set some metrics
        detector.metrics.feature_stability = 0.8
        detector.metrics.consensus_strength = 0.7
        detector.metrics.selection_entropy = 0.3
        
        initial_window_size = length(detector.sliding_window.stability_history)
        ConvergenceDetection.update_sliding_window!(detector)
        
        @test length(detector.sliding_window.stability_history) == initial_window_size + 1
        @test detector.sliding_window.stability_history[end] == 0.8
        @test detector.sliding_window.consensus_history[end] == 0.7
        @test detector.sliding_window.entropy_history[end] == 0.3
        
        # Test window size limit
        for i in 1:10
            detector.metrics.feature_stability = 0.8 + 0.01 * i
            ConvergenceDetection.update_sliding_window!(detector)
        end
        
        @test length(detector.sliding_window.stability_history) == 5  # Should be limited by window size
        
        println("  ✅ Sliding window update tests passed")
    end
    
    @testset "Threshold Adaptation Tests" begin
        # Test static threshold (no adaptation)
        static_config = create_convergence_config(threshold_adaptation = STATIC_THRESHOLD)
        detector = initialize_convergence_detector(static_config)
        
        initial_threshold = detector.current_thresholds["stability"]
        detector.stats.total_iterations = 500  # Set to trigger adaptation interval
        ConvergenceDetection.adapt_convergence_thresholds!(detector)
        
        @test detector.current_thresholds["stability"] == initial_threshold  # Should not change
        
        # Test linear adaptation
        linear_config = create_convergence_config(
            threshold_adaptation = LINEAR_ADAPTATION,
            adaptation_interval = 100,
            initial_threshold = 0.95f0,
            minimum_threshold = 0.8f0,
            maximum_iterations = 1000
        )
        detector = initialize_convergence_detector(linear_config)
        
        detector.stats.total_iterations = 500  # 50% through maximum iterations
        ConvergenceDetection.adapt_convergence_thresholds!(detector)
        
        expected_threshold = 0.95f0 - 0.5 * (0.95f0 - 0.8f0)  # Linear interpolation
        @test abs(detector.current_thresholds["stability"] - expected_threshold) < 0.01
        
        println("  ✅ Threshold adaptation tests passed")
    end
    
    @testset "Convergence Check Tests" begin
        # Test stability-based convergence
        stability_config = create_convergence_config(
            convergence_strategy = STABILITY_BASED,
            minimum_iterations = 10,
            stability_threshold = 0.9f0,
            minimum_stable_windows = 3
        )
        detector = initialize_convergence_detector(stability_config)
        
        # Set up convergence conditions
        detector.stats.total_iterations = 100
        detector.metrics.feature_stability = 0.95
        detector.sliding_window.consecutive_stable = 5
        
        converged = ConvergenceDetection.check_convergence(detector)
        @test converged == true
        
        # Test below minimum iterations
        detector.stats.total_iterations = 5
        converged = ConvergenceDetection.check_convergence(detector)
        @test converged == false
        
        # Test maximum iterations override
        detector.stats.total_iterations = 60000  # Above maximum
        converged = ConvergenceDetection.check_convergence(detector)
        @test converged == true
        
        println("  ✅ Convergence check tests passed")
    end
    
    @testset "Early Stopping Tests" begin
        # Test fixed iterations stopping
        fixed_config = create_convergence_config(
            stopping_criteria = FIXED_ITERATIONS,
            convergence_patience = 5
        )
        detector = initialize_convergence_detector(fixed_config)
        
        detector.sliding_window.consecutive_stable = 3
        should_stop = ConvergenceDetection.should_stop_early(detector)
        @test should_stop == false
        
        detector.sliding_window.consecutive_stable = 6
        should_stop = ConvergenceDetection.should_stop_early(detector)
        @test should_stop == true
        
        # Test ensemble agreement stopping
        agreement_config = create_convergence_config(
            stopping_criteria = ENSEMBLE_AGREEMENT,
            agreement_threshold = 0.9f0
        )
        detector = initialize_convergence_detector(agreement_config)
        
        detector.metrics.agreement_coefficient = 0.85
        should_stop = ConvergenceDetection.should_stop_early(detector)
        @test should_stop == false
        
        detector.metrics.agreement_coefficient = 0.95
        should_stop = ConvergenceDetection.should_stop_early(detector)
        @test should_stop == true
        
        println("  ✅ Early stopping tests passed")
    end
    
    @testset "Full Convergence Detection Cycle Tests" begin
        config = create_convergence_config(
            minimum_iterations = 5,
            stability_threshold = 0.8f0,
            minimum_stable_windows = 2
        )
        detector = initialize_convergence_detector(config)
        
        # Create mock consensus manager
        consensus_config = ConsensusVoting.create_consensus_config()
        consensus_manager = ConsensusVoting.initialize_consensus_manager(consensus_config)
        
        # Simulate convergence over multiple iterations
        for iteration in 1:10
            # Create progressively more stable feature selections
            stability_factor = min(1.0, iteration / 8.0)
            feature_selections = [
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, Int(5 + (1 - stability_factor) * 3)],
                [1, 2, 3, Int(4 + (1 - stability_factor) * 2), 5]
            ]
            
            # Add some consensus votes
            if iteration <= 3
                ConsensusVoting.cast_vote!(consensus_manager, iteration, 1, 0.8, 5)
                ConsensusVoting.build_consensus!(consensus_manager)
            end
            
            converged = ConvergenceDetection.run_convergence_detection!(
                detector, feature_selections, consensus_manager
            )
            
            if iteration >= 8  # Should converge by iteration 8
                @test converged == true
                @test detector.is_converged == true
                @test detector.convergence_confidence > 0.0
                break
            end
        end
        
        @test detector.stats.total_iterations > 0
        @test detector.stats.convergence_iterations > 0
        
        println("  ✅ Full convergence detection cycle tests passed")
    end
    
    @testset "Status and Monitoring Tests" begin
        config = create_convergence_config()
        detector = initialize_convergence_detector(config)
        
        # Initial status
        status = ConvergenceDetection.get_convergence_status(detector)
        @test haskey(status, "detector_state")
        @test haskey(status, "is_converged")
        @test haskey(status, "convergence_confidence")
        @test haskey(status, "total_iterations")
        @test haskey(status, "current_stability")
        @test haskey(status, "current_consensus")
        @test haskey(status, "current_entropy")
        @test haskey(status, "stability_threshold")
        
        @test status["detector_state"] == "active"
        @test status["is_converged"] == false
        @test status["convergence_confidence"] == 0.0
        @test status["total_iterations"] == 0
        
        # After some convergence detection activity
        detector.is_converged = true
        detector.convergence_confidence = 0.8
        detector.stats.total_iterations = 100
        detector.convergence_timestamp = now()
        
        updated_status = ConvergenceDetection.get_convergence_status(detector)
        @test updated_status["is_converged"] == true
        @test updated_status["convergence_confidence"] == 0.8
        @test updated_status["total_iterations"] == 100
        @test !isnothing(updated_status["convergence_timestamp"])
        
        println("  ✅ Status and monitoring tests passed")
    end
    
    @testset "Report Generation Tests" begin
        config = create_convergence_config()
        detector = initialize_convergence_detector(config)
        
        # Set up some state for reporting
        detector.is_converged = true
        detector.convergence_confidence = 0.9
        detector.stats.total_iterations = 500
        detector.stats.convergence_iterations = 450
        detector.stats.false_convergences = 1
        detector.stats.threshold_adaptations = 5
        detector.metrics.feature_stability = 0.95
        detector.metrics.consensus_strength = 0.88
        detector.convergence_timestamp = now()
        
        # Generate report
        report = ConvergenceDetection.generate_convergence_report(detector)
        
        @test contains(report, "Convergence Detection Report")
        @test contains(report, "Converged: Yes")
        @test contains(report, "Total Iterations: 500")
        @test contains(report, "Convergence Iterations: 450")
        @test contains(report, "Feature Stability: 0.95")
        @test contains(report, "Consensus Strength: 0.88")
        @test contains(report, "False Convergences: 1")
        @test contains(report, "Threshold Adaptations: 5")
        @test contains(report, "Strategy: $(detector.config.convergence_strategy)")
        
        println("  ✅ Report generation tests passed")
    end
    
    @testset "Reset and Cleanup Tests" begin
        config = create_convergence_config()
        detector = initialize_convergence_detector(config)
        
        # Set up some state
        detector.is_converged = true
        detector.convergence_confidence = 0.8
        detector.stats.total_iterations = 100
        detector.metrics.feature_stability = 0.9
        detector.sliding_window.consecutive_stable = 5
        
        # Add some cache entries
        detector.feature_stability_cache[[1, 2, 3]] = 0.8
        detector.analysis_times = [10.0, 20.0, 30.0]
        
        # Test reset
        ConvergenceDetection.reset_convergence_detector!(detector)
        
        @test detector.is_converged == false
        @test detector.convergence_confidence == 0.0
        @test isnothing(detector.convergence_timestamp)
        @test detector.consecutive_stable_iterations == 0
        @test detector.metrics.feature_stability == 0.0
        @test detector.sliding_window.consecutive_stable == 0
        @test detector.stats.total_iterations == 0
        @test isempty(detector.feature_stability_cache)
        
        # Test cleanup
        ConvergenceDetection.cleanup_convergence_detector!(detector)
        @test detector.detector_state == "shutdown"
        @test isempty(detector.analysis_times)
        @test isempty(detector.convergence_log)
        
        println("  ✅ Reset and cleanup tests passed")
    end
    
    @testset "Error Handling Tests" begin
        config = create_convergence_config()
        detector = initialize_convergence_detector(config)
        
        # Test with malformed feature selections
        malformed_selections = Vector{Vector{Int}}[]  # Empty vector
        stability = ConvergenceDetection.calculate_feature_stability(detector, malformed_selections)
        @test stability == 0.0
        
        # Test single feature selection
        single_selection = [[1, 2, 3]]
        stability = ConvergenceDetection.calculate_feature_stability(detector, single_selection)
        @test stability == 0.0  # Cannot calculate stability with single selection
        
        # Test convergence with insufficient history
        detector.stats.total_iterations = 100
        detector.sliding_window.consecutive_stable = 0
        converged = ConvergenceDetection.check_convergence(detector)
        @test converged == false
        
        # Test early stopping with disabled early stopping
        detector.config = create_convergence_config(early_stopping_enabled = false)
        should_stop = ConvergenceDetection.should_stop_early(detector)
        @test should_stop == false
        
        println("  ✅ Error handling tests passed")
    end
    
    @testset "Performance and Caching Tests" begin
        config = create_convergence_config(
            cache_convergence_calculations = true,
            enable_parallel_analysis = true
        )
        detector = initialize_convergence_detector(config)
        
        # Test caching performance
        large_selections = [collect(1:100) for _ in 1:10]
        
        # First calculation (should cache)
        start_time = time()
        stability1 = ConvergenceDetection.calculate_feature_stability(detector, large_selections)
        first_time = time() - start_time
        
        # Second calculation (should use cache)
        start_time = time()
        stability2 = ConvergenceDetection.calculate_feature_stability(detector, large_selections)
        second_time = time() - start_time
        
        @test stability1 == stability2
        @test second_time <= first_time  # Cache should be faster or equal
        @test length(detector.feature_stability_cache) > 0
        
        # Test analysis time tracking
        @test length(detector.analysis_times) > 0
        
        println("  ✅ Performance and caching tests passed")
    end
end

println("All Convergence Detection Algorithm tests completed!")
println("✅ Configuration and enum validation")
println("✅ Convergence metrics creation and tracking")
println("✅ Sliding window analysis implementation")
println("✅ Detector initialization and setup")
println("✅ Feature stability calculation with Jaccard similarity")
println("✅ Selection entropy calculation with Shannon entropy")
println("✅ Agreement coefficient calculation")
println("✅ Convergence metrics update and synchronization")
println("✅ Sliding window update with trend analysis")
println("✅ Adaptive threshold adjustment strategies")
println("✅ Multi-strategy convergence detection")
println("✅ Early stopping criteria and implementation")
println("✅ Complete convergence detection cycle")
println("✅ Status monitoring and detailed reporting")
println("✅ Report generation with comprehensive statistics")
println("✅ Reset and cleanup functionality")
println("✅ Error handling for edge cases")
println("✅ Performance optimization and caching mechanisms")
println("✅ Ready for MCTS ensemble convergence monitoring")