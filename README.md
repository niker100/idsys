# IDSYS: Identification Systems Framework

IDSYS is a comprehensive framework for creating, evaluating, and visualizing identification systems based on various coding schemes. It provides tools to analyze the performance, reliability, and efficiency of identification schemes across multiple parameters.

## Overview

The framework implements and evaluates different identification systems including:

- Reed-Solomon Identification (RSID)
- Concatenated Reed-Solomon Identification (RS2ID)
- Reed-Muller Identification (RMID)
- Cryptographic hash-based identification (SHA1ID, SHA256ID)

IDSYS allows systematic analysis of these systems across different parameters including Galois field sizes, message lengths, tag positions, and message patterns.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/idsys.git
cd idsys

# Install dependencies
pip install -r requirements.txt
```

## Directory Structure

```
idsys/
├── framework/           # Core framework code
│   ├── __init__.py
│   ├── core.py          # Core classes and functions
│   ├── metrics.py       # Evaluation metrics
│   ├── checkpoint.py    # Checkpointing for long analyses
│   └── utils.py         # Utility functions
├── analyses/            # Analysis scripts
│   ├── minimal_example.py
│   ├── gf_exp_influence/
│   ├── vec_length_influence/
│   ├── tag_position/
│   ├── num_messages_influence/
│   ├── collision/
│   └── ...
└── dashboard/           # Interactive visualization
    ├── dashboard_streamlit.py
    ├── dashboard_streamlit_new.py
    └── start_dashboard.md
```

## Core Concepts

### Identification Systems

An identification system in this framework consists of:

- **Encoder**: Generates tags from messages
- **Verifier**: Validates if a tag corresponds to a given message

```python
from framework import create_id_system

# Create an RS identification system
rsid = create_id_system("RSID", {"gf_exp": 8, "tag_pos": [2]})

# Encode a message
tag = rsid.send(message)

# Verify a message against a tag
is_valid = rsid.receive(tag, message)
```

### Message Generation

The framework provides utilities to generate test messages with various patterns:

```python
from framework import generate_test_messages

# Generate random messages
messages = generate_test_messages(vec_len=16, gf_exp=8, count=100)

# Generate structured messages with specific patterns
from framework.core import generate_structured_messages

structured_msgs = generate_structured_messages(
    vec_len=16,
    pattern_type="repeated_patterns",  # Options: random, incremental, repeated_patterns, sparse, low_entropy, only_two
    gf_exp=8,
    target_count=100
)
```

### Metrics and Evaluation

The framework includes comprehensive metrics for system evaluation:

```python
from framework import IdMetrics

# Evaluate a single system
metrics = IdMetrics.evaluate_system(
    system=rsid,
    vec_len=16,
    num_messages=10000
)

# Compare multiple systems
system_comparison = IdMetrics.compare_systems(
    systems={"RSID": rsid, "SHA1ID": sha1id},
    num_messages=10000,
    vec_len=16
)
```

### Checkpointing for Long Analyses

For long-running analyses, the framework provides checkpointing:

```python
from framework.checkpoint import create_checkpoint_manager

# Create checkpoint manager
checkpoint = create_checkpoint_manager(
    output_dir="output/my_analysis",
    analysis_name="parameter_sweep",
    save_interval=10
)

# Initialize with all parameter sets to test
remaining_params = checkpoint.initialize_analysis(parameter_sets)

# Process each parameter combination
for params in remaining_params:
    result = analyze_single_combination(params)
    checkpoint.add_result(params, result)

# Finalize and get results
checkpoint.finalize_analysis()
results_df = checkpoint.get_results_dataframe()
```

## Running Analyses

The analyses directory contains scripts for evaluating different aspects of identification systems:

```bash
# Run minimal example
python analyses/minimal_example.py

# Run GF exponent influence analysis with checkpointing
python analyses/gf_exp_influence/gf_exp_checkpointed.py

# Run tag position influence analysis
python analyses/tag_position/tag_position.py
```

## Interactive Dashboard

IDSYS includes a Streamlit dashboard for visualizing results:

```bash
# Navigate to dashboard directory
cd dashboard

# Start the dashboard
streamlit run dashboard_streamlit.py
```

The dashboard provides interactive exploration of:
- False positive rates for identification systems
- Execution time comparisons
- System type comparisons
- Probability distribution functions & examples

## Example Analyses

### Comprehensive Benchmark

```bash
# Run comprehensive benchmark of all systems
python analyses/big_script/big_script.py
```

This script evaluates all identification systems across:
- Multiple system types (RSID, RMID, SHA1ID, etc.)
- Various Galois field exponents (8, 16, 32, 64)
- Different vector lengths
- Multiple message patterns

### Collision Analysis

```bash
# Run collision analysis
python analyses/collision/test.py
```

Analyzes collision behavior between random and structured messages across different system types.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This framework utilizes the `ecidcodes` library for implementing Reed-Solomon and Reed-Muller identification codes.

Similar code found with 2 license types