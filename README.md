# Identification System Framework

A comprehensive Python framework for creating, evaluating, and visualizing identification systems. This framework provides tools for implementing various identification coding schemes, measuring their performance, and analyzing the results through visualizations.

## Overview

Identification systems are communication systems where a sender (Alice) encodes a message, and a receiver (Bob) needs to determine whether the received codeword corresponds to a specific message. This differs from traditional communication where Bob needs to decode which message was sent.

This framework implements:

1. Different identification system encoding schemes
2. Metrics for evaluating performance
3. Visualization tools for analysis
4. Testing utilities

## Structure

The framework consists of the following components:

- `core.py`: Base classes and implementations for identification systems
- `metrics.py`: Functions for measuring system performance
- `visualization.py`: Tools for creating visual representations of results
- `test_identification.py`: Test suite for the framework

## Usage

### Creating an Identification System

```python
from framework import create_id_system, generate_string_messages

# Create a hash-based identification system
hash_system = create_id_system("hash_tagging", {"code_length": 16})

# Create a random projection system with a distance threshold
rp_system = create_id_system("random_projection", {
    "code_length": 16,
    "max_distance": 2,
    "seed": 42
})

# Generate test messages
messages = generate_string_messages(count=10, length=8)
```

### Evaluating System Performance

```python
from framework import IdMetrics

# Calculate reliability
reliability = IdMetrics.reliability(hash_system, messages, num_trials=1000)
print(f"System reliability: {reliability:.4f}")

# Calculate error rates
error_rates = IdMetrics.error_rates(hash_system, messages, num_trials=500)
print(f"False positive rate: {error_rates['false_positive_rate']:.4f}")
print(f"False negative rate: {error_rates['false_negative_rate']:.4f}")

# Calculate worst-case collision probability
collision_prob = IdMetrics.worst_case_collision_probability(
    hash_system, messages, sample_size=10, num_trials=100
)
print(f"Worst-case collision probability: {collision_prob:.4f}")

# Calculate efficiency metrics
efficiency = IdMetrics.efficiency(hash_system)
print(f"Code rate: {efficiency['code_rate']:.4f}")
print(f"Encoding time: {efficiency['encoding_time_ms']:.4f} ms")
```

### Creating Visualizations

```python
from framework import IdVisualizer
import matplotlib.pyplot as plt

# Plot reliability vs code length
code_lengths = [4, 8, 12, 16, 24, 32]
fig, ax = IdVisualizer.plot_reliability_vs_code_length(
    hash_system, messages, code_lengths, num_trials=100
)
plt.show()

# Create a comprehensive dashboard
fig = IdVisualizer.create_dashboard(
    hash_system, messages, code_lengths, num_trials=100
)
plt.savefig('system_dashboard.png')
plt.show()
```

## Running the Example Script

The `example_identification.py` script demonstrates how to use the framework:

```bash
python example_identification.py
```

This will run a comprehensive analysis of different identification systems, including:
- Comparison of multiple identification schemes
- Parameter effect analysis
- Noise robustness testing
- Creation of visualization dashboards

The example script will save visualization results as PNG files.

## Components

### Encoders

- **HashTaggingEncoder**: Uses cryptographic hash functions for encoding messages
- **RandomProjectionEncoder**: Uses random projections for approximate matching

### Decoders

- **BitwiseCompareDecoder**: Compares codewords bitwise with a threshold
- **HammingDistanceDecoder**: Uses Hamming distance for identification decisions

### Metrics

- **Reliability**: Probability of correct identification
- **Error Rates**: False positive and false negative rates
- **Collision Probability**: Likelihood of messages being confused
- **Efficiency**: Code rate and encoding time

### Visualization Tools

- Individual metric plots
- Parameter sweep visualizations
- Comprehensive dashboards
- Comparison between systems

## Running Tests

```bash
python -m unittest framework.test_identification
```

## References

For more information about identification coding, refer to the literature in the `literature/` folder.
