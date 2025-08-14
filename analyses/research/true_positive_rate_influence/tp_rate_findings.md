# True Positive Rate Influence Analysis - Findings

## Overview
This analysis examined how varying the true positive rate (probability of positive identification scenarios) affects the reliability and false positive rates of three identification systems: RSID, RMID, and SHA1ID using 1,000,000 trials per data point.

## Methodology
- **Systems Tested**: Reed-Solomon ID (RSID), Reed-Muller ID (RMID), and SHA1-based ID (SHA1ID)
- **Parameters**: All systems configured with GF exponent = 8, message length = 16 bytes, 10000 messages
- **True Positive Rate Range**: 0.0 to 1.0 in 0.2 increments
- **Test Protocol**: Based on `_calculate_reliability_and_fp_rate()` method from metrics.py

## Definitions from our Framework

### Reliability Calculation
From the metrics code, reliability is defined as:
```
reliability = correct / num_trials
```
Where `correct` includes both:
- Correct positive identifications (when `system.receive(codeword, msg) == True` for matching pairs)
- Correct negative identifications (when `system.receive(codeword, different_msg) == False` for non-matching pairs)

### False Positive Rate Calculation
```
fp_rate = false_positives / true_negatives
```
Where `false_positives` occur when `system.receive(codeword, different_msg) == True` (incorrect acceptance).

## Experimental Results

### 1. Reliability Performance
**Linear relationship with true positive rate:**

- **All systems** show remarkably similar performance:
  - Average reliability: 0.9980 (99.8%)
  - Range: ~0.9960 (p=0.0) to 1.0000 (p=1.0)
  - Slope: 0.0039 (0.39%) reliability improvement per unit true positive rate increase

### 2. False Positive Rate Analysis
**Constant FPR values independent of true positive rate:**

- **RSID**: FPR ≈ 0.003921 (0.39%)
- **RMID**: FPR ≈ 0.003961 (0.40%)  
- **SHA1ID**: FPR ≈ 0.003981 (0.40%)

All systems show FPR values very close to the theoretical 2^-8 ≈ 0.00390625 (0.39%).

**Observation**: The drop to 0% FPR at p=1.0 occurs because no negative identification scenarios exist when true positive rate = 1.0.

## Discoveries

### Linear Relationship
The experimental data confirms the relationship:

**`(reliability - 1) = false_positive_rate × (true_positive_rate - 1)`**

#### Theoretical Verification:
Let:
- `p` = true positive rate
- `FPR` = false positive rate (constant for each system)
- assume there are no false negatives leads to:
```
reliability = p × 1.0 + (1-p) × (1-FPR) = 1 - (1-p) × FPR
```

Rearranging:
```
reliability = 1 - FPR + p × FPR
reliability - 1 = FPR × (p - 1)
(reliability - 1) = FPR × (true_positive_rate - 1)
```

Furthermore, we get:
```
d(reliability)/d(p) = FPR
```


This confirms the observed linear relationship and validates the experiment.

#### Experimental Findings
The slopes in reliability plots exactly match the false positive rates:
- All systems: slope ≈ 0.0039 = FPR ≈ 2^-8

This validates our derived relationship: `d(reliability)/d(p) = FPR`

## System Comparison

### Cryptographic vs. Coding Theory Approaches
With GF exponent = 8 configuration, all three systems demonstrate nearly identical performance:

1. **SHA1ID** (cryptographic hash): FPR ≈ 0.003981
2. **RMID** (Reed-Muller): FPR ≈ 0.003961
3. **RSID** (Reed-Solomon): FPR ≈ 0.003921

These results suggest that with this parameter configuration, the algebraic properties of the coding-based systems (RSID, RMID) can achieve performance comparable to cryptographic approaches.

## Implications for Metrics Usage

### 1. Scenario-Dependent Reliability Assessment
**Key insight**: Reliability is not an intrinsic system property but depends on the application's positive/negative identification ratio.

**Practical implications**:
- **Authentication systems** (high positive rate): All systems perform similarly (>99.9%)
- **Intrusion detection** (low positive rate): All systems maintain good reliability (≈99.6%)
- **Mixed workloads**: Performance can be predicted using `reliability = 1 - FPR × (1-p)`

### 2. False Positive Rate as Primary Metric
**FPR emerges as the fundamental system characteristic**:
- Independent of scenario distribution
- Directly determines reliability sensitivity to workload changes
- More suitable for system comparison and selection

### 3. Design Trade-offs
The mathematical relationship reveals fundamental trade-offs:
```
Δ(reliability) = FPR × Δ(true_positive_rate)
```

Systems with lower FPR are:
- More stable across different workloads
- Better suited for variable or unknown scenario distributions
- Preferable for high-security applications