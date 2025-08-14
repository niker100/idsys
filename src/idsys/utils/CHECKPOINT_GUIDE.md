# Checkpoint Guide

This guide explains the approaches for saving analysis data periodically and resuming computations after interruptions in your identification systems analysis framework.

## Table of Contents
1. [Quick Start: Using the Checkpointing Framework](#quick-start)
2. [Integration Examples](#examples)
3. [Migration from Existing Scripts](#migration)
4. [Advanced Features](#advanced)
5. [Troubleshooting](#troubleshooting)

## Quick Start: Using the Checkpointing Framework {#quick-start}

The easiest way to add resumable computation to your analysis scripts is using the provided checkpointing framework:

```python
from idsys import create_checkpoint_manager

# Create checkpoint manager
checkpoint = create_checkpoint_manager(
    output_dir="my_analysis_output",
    analysis_name="my_analysis",
    save_interval=10  # Save every 10 iterations
)

# Define your parameter combinations
parameter_sets = [
    {"gf_exp": 8, "system_type": "RSID", "vec_len": 125},
    {"gf_exp": 16, "system_type": "RSID", "vec_len": 125},
    # ... more parameters
]

# Initialize (resumes from checkpoint if available)
remaining_params = checkpoint.initialize_analysis(parameter_sets)

# Process each parameter set
for params in remaining_params:
    results = run_my_analysis(params)  # Your analysis function
    checkpoint.add_result(params, results)

# Finalize
checkpoint.finalize_analysis()
```

## Integration Examples {#examples}

### Example 1: Minimal Integration

Convert your existing loop with minimal changes:

```python
# Before (your existing code)
for gf_exp in gf_exp_values:
    for system_type in system_types:
        system = create_id_system(system_type, {"gf_exp": gf_exp})
        metrics = evaluate_system(system)
        # Results were lost if script crashed here

# After (with checkpointing)
from idsys import create_checkpoint_manager

checkpoint = create_checkpoint_manager("output", "my_analysis")

parameter_sets = [
    {"gf_exp": gf_exp, "system_type": system_type}
    for gf_exp in gf_exp_values
    for system_type in system_types
]

remaining = checkpoint.initialize_analysis(parameter_sets)

for params in remaining:
    system = create_id_system(params["system_type"], {"gf_exp": params["gf_exp"]})
    metrics = evaluate_system(system)
    checkpoint.add_result(params, metrics)

checkpoint.finalize_analysis()
```

### Example 2: Using the Quick Wrapper

For even simpler integration:

```python
from idsys.utils.migration_utils import quick_checkpoint_wrapper

def analyze_single_combination(params):
    """Your analysis logic for one parameter combination."""
    system = create_id_system(params["system_type"], {"gf_exp": params["gf_exp"]})
    return evaluate_system(system)

# Run with automatic checkpointing
results_df = quick_checkpoint_wrapper(
    analysis_function=analyze_single_combination,
    parameter_sets=parameter_sets,
    output_dir="my_analysis",
    analysis_name="system_comparison"
)
```

### Example 3: Manual CSV Saving (Simplest Approach)

If you prefer a simple manual approach:

```python
import pandas as pd
import os

# Load existing results if available
csv_file = "my_results.csv"
if os.path.exists(csv_file):
    results_df = pd.read_csv(csv_file)
    completed_params = set(results_df[['gf_exp', 'system_type']].apply(tuple, axis=1))
else:
    results_df = pd.DataFrame()
    completed_params = set()

# Process only new parameters
all_results = []
for gf_exp in gf_exp_values:
    for system_type in system_types:
        if (gf_exp, system_type) in completed_params:
            continue  # Skip already completed
        
        # Run analysis
        results = run_analysis(gf_exp, system_type)
        results['gf_exp'] = gf_exp
        results['system_type'] = system_type
        all_results.append(results)
        
        # Save every 5 iterations
        if len(all_results) % 5 == 0:
            new_df = pd.DataFrame(all_results)
            results_df = pd.concat([results_df, new_df], ignore_index=True)
            results_df.to_csv(csv_file, index=False)
            all_results = []  # Clear batch

# Final save
if all_results:
    new_df = pd.DataFrame(all_results)
    results_df = pd.concat([results_df, new_df], ignore_index=True)
    results_df.to_csv(csv_file, index=False)
```

## Migration from Existing Scripts {#migration}

### Step 1: Convert JSON Results to CSV

```python
from idsys.utils.migration_utils import convert_json_to_csv

# Convert your existing JSON files
convert_json_to_csv("analyses/gf_exp_influence/system_results.json")
convert_json_to_csv("analyses/num_messages_influence/system_results.json")
```

### Step 2: Restructure Analysis Loop

Transform your existing analysis pattern:

```python
# Original pattern
system_results = {name: {'metric1': [], 'metric2': []} for name in systems}

for param in parameters:
    for system_name, system in systems.items():
        metrics = evaluate(system, param)
        system_results[system_name]['metric1'].append(metrics['metric1'])
        system_results[system_name]['metric2'].append(metrics['metric2'])

# Save at the end (data lost if crash occurs)
with open('results.json', 'w') as f:
    json.dump(system_results, f)
```

```python
# Checkpointed pattern
parameter_sets = [
    {"param": param, "system_name": system_name}
    for param in parameters
    for system_name in systems.keys()
]

checkpoint = create_checkpoint_manager("output", "analysis")
remaining = checkpoint.initialize_analysis(parameter_sets)

for params in remaining:
    system = systems[params["system_name"]]
    metrics = evaluate(system, params["param"])
    checkpoint.add_result(params, metrics)  # Auto-saved periodically

checkpoint.finalize_analysis()
```

## Advanced Features {#advanced}

### Custom Save Intervals

```python
# Save after every parameter combination (safest)
checkpoint = create_checkpoint_manager("output", "analysis", save_interval=1)

# Save every 10 combinations (faster)
checkpoint = create_checkpoint_manager("output", "analysis", save_interval=10)

# Save every 100 combinations (fastest, but higher risk)
checkpoint = create_checkpoint_manager("output", "analysis", save_interval=100)
```

### Custom Backup Management

```python
# Keep more backup files
checkpoint = AnalysisCheckpoint(
    output_dir="output",
    analysis_name="analysis",
    save_interval=10,
    backup_count=10  # Keep 10 backup versions
)
```

### Progress Monitoring

```python
# Get progress information
completion = checkpoint.get_completion_percentage()
print(f"Analysis {completion:.1f}% complete")

# Check if specific parameter was completed
if checkpoint.is_parameter_completed({"gf_exp": 8, "system": "RSID"}):
    print("This combination was already processed")
```

### Working with Results

```python
# Get results as DataFrame for analysis
df = checkpoint.get_results_dataframe()

# Group by system type
for system_type, group in df.groupby('system_type'):
    print(f"{system_type}: {len(group)} results")

# Plot results
import matplotlib.pyplot as plt
for system_type in df['system_type'].unique():
    system_data = df[df['system_type'] == system_type]
    plt.plot(system_data['gf_exp'], system_data['false_positive_rate'], 
             label=system_type, marker='o')

plt.xlabel('GF Exponent')
plt.ylabel('False Positive Rate')
plt.legend()
plt.savefig('analysis_results.png')
```

## Troubleshooting {#troubleshooting}

### Common Issues and Solutions

**1. Import errors with checkpoint module**
```bash
# Make sure you're in the correct directory
cd /workspaces/idsys
python -c "from idsys import create_checkpoint_manager"
```

**2. CSV file corruption**
The checkpoint system automatically creates backups. If your main CSV gets corrupted:
```python
# Look in the backups directory
ls analyses/my_analysis/backups/
# Restore from the latest backup
cp analyses/my_analysis/backups/my_analysis_results_20250623_143022.csv analyses/my_analysis/my_analysis_results.csv
```

**3. Memory issues with large datasets**
```python
# Use smaller save intervals to reduce memory usage
checkpoint = create_checkpoint_manager("output", "analysis", save_interval=1)

# Or process in smaller batches
batch_size = 100
for i in range(0, len(parameter_sets), batch_size):
    batch = parameter_sets[i:i+batch_size]
    # Process batch...
```

**4. Resuming with different parameter sets**
If you need to add more parameters to an existing analysis:
```python
# Load existing checkpoint
checkpoint = create_checkpoint_manager("output", "analysis")

# Add new parameters to existing ones
all_parameters = original_parameters + new_parameters
remaining = checkpoint.initialize_analysis(all_parameters)
# Will only process the new parameters
```

### File Structure

When using the checkpointing system, you'll get this file structure:

```
analyses/
  my_analysis/
    my_analysis_results.csv          # Main results file
    my_analysis_metadata.json        # Analysis configuration
    my_analysis_progress.json        # Progress tracking
    my_analysis_checkpoint.log       # Execution log
    backups/                         # Automatic backups
      my_analysis_results_20250623_143022.csv
      my_analysis_metadata_20250623_143022.json
      ...
```

### Performance Tips

1. **Choose appropriate save intervals**: Balance between safety (low interval) and performance (high interval)
2. **Use SSD storage**: Checkpoint files benefit from fast disk I/O
3. **Monitor disk space**: Keep an eye on backup accumulation
4. **Consider compression**: For very large datasets, consider using parquet format instead of CSV

## Summary

For your identification systems analysis framework, I recommend:

1. **Start with the CSV-based checkpointing system** - it's specifically designed for your use case
2. **Use the quick wrapper for existing scripts** - minimal code changes required
3. **Set save_interval=1 for long-running analyses** - save after each parameter combination
4. **Convert existing JSON files to CSV** for consistency
5. **Consider SQLite only if you need concurrent access** from multiple scripts

The checkpointing framework provides automatic resume capability, progress tracking, and data safety with minimal changes to your existing code.
