# IDSYS Core Module

This module provides the fundamental classes and functions for implementing and evaluating identification systems with multiple coding schemes.

## Overview

The core module implements a clean architecture pattern with:
- Abstract base classes for encoders and verifiers
- Concrete implementations for different identification systems
- A unified `IdSystem` interface
- Utility functions for message generation and testing
- Comprehensive metrics evaluation tools

## Supported Systems

| System | Description | Parameters |
|--------|-------------|------------|
| RSID | Reed-Solomon Identification | `gf_exp`, `tag_pos` |
| RS2ID | Concatenated Reed-Solomon ID | `gf_exp`, `tag_pos`, `tag_pos_in` |
| RMID | Reed-Muller Identification | `gf_exp`, `tag_pos`, `rm_order` |
| SHA1ID | SHA1-based Identification | `gf_exp` |
| SHA256ID | SHA256-based Identification | `gf_exp` |
| NoCode | Baseline system for testing | `gf_exp` |

## Quick Start

```python
from idsys import create_id_system, generate_test_messages

# Create an RSID system
system = create_id_system("RSID", {
    "gf_exp": 8,
    "tag_pos": [2, 3]  # Multi-tag system
})

# Generate test messages
messages = generate_test_messages(vec_len=16, gf_exp=8, count=10)

# Encode and verify
for message in messages:
    tag = system.send(message)
    is_valid = system.receive(tag, message)
    print(f"Message: {message[:4]}... -> Tag: {tag} -> Valid: {is_valid}")
```

## Architecture

```
IdSystem
├── IdEncoder (abstract)
│   ├── RSIDEncoder
│   ├── RS2IDEncoder  
│   ├── RMIDEncoder
│   ├── SHA1IDEncoder
│   ├── SHA256IDEncoder
│   └── NoCodeEncoder
└── IdVerifier (abstract)
    ├── RSIDVerifier
    ├── RS2IDVerifier
    ├── RMIDVerifier  
    ├── SHA1IDVerifier
    ├── SHA256IDVerifier
    └── NoCodeVerifier
```

## Message Generation

The module provides two message generation functions:

1. **`generate_test_messages()`** - Random messages for general testing
2. **`generate_structured_messages()`** - Patterned messages for specific tests

Supported patterns:
- `random`: Truly random messages using idcodes
- `incremental`: Sequential patterns
- `repeated_patterns`: Repeating byte sequences
- `sparse`: Mostly zeros with few non-zero elements
- `low_entropy`: Limited alphabet messages
- `only_two`: Binary-like patterns

## Metrics and Evaluation

The module includes comprehensive metrics for evaluating identification systems:

```python
from idsys import create_id_system
from idsys import IdMetrics

# Create a system to evaluate
system = create_id_system("RSID", {
    "gf_exp": 8,
    "tag_pos": [2]
})

# Evaluate the system
metrics = IdMetrics.evaluate_system(
    system,
    num_messages=10000,
    vec_len=16,
    message_pattern='random'
)

# Print key metrics
print(f"False positive rate: {metrics['false_positive_rate']:.8f}")
print(f"Code rate: {metrics['code_rate_bulk']:.6f}")
print(f"Average execution time: {metrics['avg_execution_time_ms']:.3f} ms")
print(f"Throughput: {metrics['throughput_msgs_per_sec']:.1f} msgs/sec")
```

Metrics include:
- Performance: false positive rates, reliability
- Timing: execution time, throughput
- Code characteristics: code rate, tag size
- Message properties: Hamming distances, symbol distributions
- Statistical analysis: theoretical vs. observed metrics

## Dependencies

- `ecidcodes`: Core identification algorithms library
- `numpy`: Numerical operations
- `typing`: Type hints support
- `multiprocessing`: Parallel evaluation support