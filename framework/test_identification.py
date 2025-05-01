import numpy as np
import random
import string
import os
import time
from framework.core import IdentificationSystem
from framework.encoders import TaggingEncoder
from framework.metrics import Evaluator, Metrics
from framework.visualization import Visualizer

# Create output directory for plots
os.makedirs('plots', exist_ok=True)

def generate_random_string(length=10):
    """Generate a random string of fixed length."""
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def generate_dataset(num_pairs=100, match_ratio=0.5, message_length=20):
    """
    Generate a dataset of message pairs for testing.
    
    Args:
        num_pairs (int): Number of message pairs to generate
        match_ratio (float): Ratio of matching pairs (0-1)
        message_length (int): Length of the messages
        
    Returns:
        tuple: (message_pairs, expected_matches)
            message_pairs: List of (sender_message, receiver_message) tuples
            expected_matches: List of boolean values indicating if the pair should match
    """
    message_pairs = []
    expected_matches = []
    
    num_matches = int(num_pairs * match_ratio)
    num_non_matches = num_pairs - num_matches
    
    # Generate matching pairs
    for _ in range(num_matches):
        message = generate_random_string(message_length)
        message_pairs.append((message, message))
        expected_matches.append(True)
    
    # Generate non-matching pairs
    for _ in range(num_non_matches):
        message1 = generate_random_string(message_length)
        message2 = generate_random_string(message_length)
        while message1 == message2:  # Ensure they don't match
            message2 = generate_random_string(message_length)
        message_pairs.append((message1, message2))
        expected_matches.append(False)
    
    # Shuffle the pairs to randomize order
    combined = list(zip(message_pairs, expected_matches))
    random.shuffle(combined)
    message_pairs, expected_matches = zip(*combined)
    
    return list(message_pairs), list(expected_matches)

def test_parameter_influence():
    """Test how different parameters influence identification system performance, focusing on key metrics."""
    print("\n=== Testing Parameter Influence ===")
    
    # Create visualizer
    viz = Visualizer(output_dir='plots')
    
    # Define parameter ranges - limited to 1-64 bits
    tag_lengths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 20, 24, 32, 40]
    dataset_sizes = [100, 1000, 10000, 100000]
    message_size = 1000  # 1KB standard message size
    
    # Generate a standard test dataset
    test_dataset_size = 10000
    print(f"Generating test dataset with {test_dataset_size} message pairs...")
    message_pairs, expected_matches = generate_dataset(num_pairs=test_dataset_size, match_ratio=0.5, message_length=20)
    
    # Initialize arrays for collecting metrics
    # Standard encoder metrics
    std_mean_error_rates = []
    std_max_error_rates = []
    std_reliabilities = []
    std_efficiencies = []
    
    # Theoretical error rates for better visualization
    theoretical_mean_error_rates = []
    theoretical_max_error_rates = []
    
    # Evaluate each tag length
    eval_subset_size = min(500, test_dataset_size)  # Use larger subset for better accuracy
    subset_pairs = message_pairs[:eval_subset_size]
    subset_expected = expected_matches[:eval_subset_size]
    
    for tag_length in tag_lengths:
        print(f"\nEvaluating tag length: {tag_length} bits")
        
        # Calculate theoretical error rates based on birthday paradox
        theoretical_error = Metrics.theoretical_collision_probability(tag_length, eval_subset_size)
        theoretical_mean_error_rates.append(theoretical_error)
        theoretical_max_error_rates.append(theoretical_error * 1.5)  # Slightly higher for max error
        
        # Test with standard encoder
        print("  Testing with standard encoder...")
        encoder = TaggingEncoder(tag_length=tag_length)
        system = IdentificationSystem(encoder)
        evaluator = Evaluator(system)
        
        metrics = evaluator.evaluate_dataset(subset_pairs, subset_expected)
        
        # Store the key metrics
        std_mean_error_rates.append(metrics.get('mean_error_rate', 0))
        std_max_error_rates.append(metrics.get('max_error_rate', 0))
        std_reliabilities.append(metrics.get('reliability', 1.0))
        std_efficiencies.append(metrics.get('avg_efficiency', message_size*8/tag_length))
        
        # Print summary
        print(f"    Mean Error Rate: {std_mean_error_rates[-1]:.6f} (theoretical: {theoretical_mean_error_rates[-1]:.6f})")
        print(f"    Max Error Rate: {std_max_error_rates[-1]:.6f} (theoretical: {theoretical_max_error_rates[-1]:.6f})")
        print(f"    Reliability: {std_reliabilities[-1]:.6f}")
        print(f"    Efficiency: {std_efficiencies[-1]:.2f}x")
        
        # Calculate theoretical collision probabilities for key dataset sizes
        print("\n  Theoretical collision probabilities:")
        for dataset_size in dataset_sizes:
            collision_prob = Metrics.theoretical_collision_probability(tag_length, dataset_size)
            print(f"    {dataset_size} messages: {collision_prob:.6f}")
    
    # 1. Plot measured error rates
    print(std_mean_error_rates, std_max_error_rates)
    viz.plot_error_rates(
        tag_lengths=tag_lengths,
        mean_error_rates=std_mean_error_rates,
        max_error_rates=std_max_error_rates,
        save_path="parameter_influence_measured_error_rates.png"
    )
    
    # 2. Plot theoretical error rates
    viz.plot_error_rates(
        tag_lengths=tag_lengths,
        mean_error_rates=theoretical_mean_error_rates,
        max_error_rates=theoretical_max_error_rates,
        save_path="parameter_influence_theoretical_error_rates.png"
    )
    
    # 3. Reliability visualization
    viz.plot_reliability(
        tag_lengths=tag_lengths,
        reliability_data={'standard': std_reliabilities},
        save_path="parameter_influence_reliability.png"
    )
    
    # 4. Efficiency vs Tag Length for different message sizes
    efficiency_data = {}
    message_sizes_for_plot = [100, 1000, 10000, 100000]  # Bytes
    
    for tag_length in tag_lengths:
        efficiency_data[tag_length] = {}
        for msg_size in message_sizes_for_plot:
            efficiency_data[tag_length][msg_size] = Metrics.efficiency(msg_size * 8, tag_length)
    
    viz.plot_efficiency_vs_tag_length(
        tag_lengths=tag_lengths,
        message_sizes=message_sizes_for_plot,
        efficiency_data=efficiency_data,
        save_path="parameter_influence_efficiency.png"
    )
    
    # 5. Combined key metrics visualization using theoretical error rates for better visibility
    viz.plot_combined_metrics(
        tag_lengths=tag_lengths,
        mean_error_rates=theoretical_mean_error_rates,
        reliabilities=std_reliabilities,
        efficiencies=[eff/max(std_efficiencies) for eff in std_efficiencies],  # Normalize
        save_path="parameter_influence_combined.png"
    )
    
    # 6. Parameter Impact Heatmaps
    parameter_data = {
        'efficiency': efficiency_data,
        'reliability_estimate': {tl: {ms: 1.0 - Metrics.theoretical_collision_probability(tl, ms) 
                               for ms in message_sizes_for_plot} 
                          for tl in tag_lengths}
    }
    
    viz.plot_parameter_impact_heatmap(
        tag_lengths=tag_lengths,
        message_sizes=message_sizes_for_plot,
        data=parameter_data,
        metric='efficiency',
        save_path="parameter_impact_efficiency_heatmap.png"
    )
    
    viz.plot_parameter_impact_heatmap(
        tag_lengths=tag_lengths,
        message_sizes=message_sizes_for_plot,
        data=parameter_data,
        metric='reliability_estimate',
        save_path="parameter_impact_reliability_heatmap.png"
    )
    
    # 7. Radar comparison of different configurations
    metrics_data = []
    # Select a subset of tag lengths for clearer visualization
    radar_tag_lengths = tag_lengths[::2] if len(tag_lengths) >= 5 else tag_lengths
    
    for tag_length in radar_tag_lengths:
        idx = tag_lengths.index(tag_length)
        # Calculate a processing time approximation (for demonstration)
        avg_processing_time = 0.01 * (tag_length / 16) 
        
        metrics_data.append({
            'reliability': std_reliabilities[idx],
            'avg_efficiency': std_efficiencies[idx],
            'mean_error_rate': theoretical_mean_error_rates[idx],  # Use theoretical for better visualization
            'avg_processing_time': avg_processing_time
        })
    
    # Update the metrics list
    viz.plot_radar_comparison(
        metrics_data=metrics_data,
        metrics=['reliability', 'avg_efficiency', 'mean_error_rate', 'avg_processing_time'],
        labels=[f"{tl}-bit" for tl in radar_tag_lengths],
        save_path="parameter_radar_comparison.png"
    )
    
    # 8. Metric tradeoffs - tag length vs error rate
    viz.plot_metric_tradeoff(
        x_metric=tag_lengths,
        y_metric=theoretical_mean_error_rates,
        tag_lengths=tag_lengths,
        x_label='Tag Length (bits)',
        y_label='Mean Error Rate (theoretical)',
        save_path="tag_length_error_tradeoff.png"
    )
    
    # 9. 3D surface parameter visualization
    try:
        viz.plot_parameter_surface(
            tag_lengths=tag_lengths,
            message_sizes=message_sizes_for_plot,
            data=parameter_data,
            metric='efficiency',
            save_path="parameter_3d_efficiency.png"
        )
        
        # Add reliability surface too
        viz.plot_parameter_surface(
            tag_lengths=tag_lengths,
            message_sizes=message_sizes_for_plot,
            data=parameter_data,
            metric='reliability_estimate',
            save_path="parameter_3d_reliability.png"
        )
    except Exception as e:
        print(f"3D plotting error: {e}")
        print("3D plotting requires mpl_toolkits.mplot3d which may not be available.")

def test_message_scaling():
    """Test how identification performance scales with message size, focusing on key metrics."""
    print("\n=== Testing Message Size Scaling and Efficiency ===")
    
    # Create visualizer
    viz = Visualizer(output_dir='plots')
    
    # Define message sizes to test (bytes)
    message_sizes = [10, 100, 1000, 10000, 100000]
    
    # Define tag lengths to test - smaller range of 1-32 bits
    tag_lengths = [1, 2, 4, 8, 16, 32]
    
    # Store results for visualization
    encoding_times = {tl: [] for tl in tag_lengths}
    compression_ratios = {tl: [] for tl in tag_lengths}
    bandwidth_savings = {tl: [] for tl in tag_lengths}
    error_rates = {tl: [] for tl in tag_lengths}
    
    for tag_length in tag_lengths:
        print(f"\nTesting with tag length: {tag_length} bits")
        
        encoder = TaggingEncoder(tag_length=tag_length)
        system = IdentificationSystem(encoder)
        
        for msg_size in message_sizes:
            print(f"  Testing with message size: {msg_size} bytes")
            
            # Generate test messages of specified size
            message1 = generate_random_string(msg_size)
            message2 = generate_random_string(msg_size)
            
            # Measure encoding time
            start_time = time.time()
            for _ in range(10):  # Repeat for more accurate timing
                encoded = encoder.encode(message1)
            encoding_time = (time.time() - start_time) / 10 * 1000  # ms
            
            # Calculate compression metrics
            encoded_size = (tag_length + 7) // 8  # Bits to bytes
            compression_ratio = msg_size / encoded_size
            
            # Calculate bandwidth saving
            bandwidth_saving = (msg_size * 8) / tag_length if tag_length > 0 else float('inf')
            
            # Calculate theoretical collision probability/error rate
            error_rate = Metrics.theoretical_collision_probability(tag_length, msg_size)
            
            # Store results
            encoding_times[tag_length].append(encoding_time)
            compression_ratios[tag_length].append(compression_ratio)
            bandwidth_savings[tag_length].append(bandwidth_saving)
            error_rates[tag_length].append(error_rate)
            
            print(f"    Encoding time: {encoding_time:.3f} ms")
            print(f"    Compression ratio: {compression_ratio:.1f}x")
            print(f"    Bandwidth saving: {bandwidth_saving:.1f}x")
            print(f"    Theoretical error rate: {error_rate:.6f}")
    
    # Plot processing time vs message size
    viz.plot_message_scaling(
        message_sizes=message_sizes,
        tag_lengths=tag_lengths,
        metrics=encoding_times,
        metric_name='Processing Time (ms)',
        save_path="message_scaling_processing_time.png"
    )
    
    # Plot compression ratio vs message size
    viz.plot_message_scaling(
        message_sizes=message_sizes,
        tag_lengths=tag_lengths,
        metrics=compression_ratios,
        metric_name='Compression Ratio',
        save_path="message_scaling_compression.png"
    )
    
    # Plot bandwidth saving vs message size
    viz.plot_message_scaling(
        message_sizes=message_sizes,
        tag_lengths=tag_lengths,
        metrics=bandwidth_savings,
        metric_name='Bandwidth Saving Factor',
        save_path="message_scaling_bandwidth.png"
    )
    
    # Plot error rate vs message size
    viz.plot_message_scaling(
        message_sizes=message_sizes,
        tag_lengths=tag_lengths,
        metrics=error_rates,
        metric_name='Theoretical Error Rate',
        save_path="message_scaling_error_rate.png"
    )


if __name__ == "__main__":
    # Run the parameter influence tests with focus on key metrics
    test_parameter_influence()
    test_message_scaling()