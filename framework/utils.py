import numpy as np
from typing import List, Dict, Any
from .core import IdSystem, generate_test_messages
from .metrics import IdMetrics


# Additional utility functions for specific use cases
def evaluate_system_with_generated_messages(
    system: IdSystem,
    vec_len: int,
    gf_exp: int,
    num_messages: int = 100,
    **kwargs
) -> Dict[str, float]:
    """
    Evaluate a system with automatically generated test messages.
    
    Args:
        system: The identification system to evaluate
        vec_len: Vector length for message generation
        gf_exp: Galois field exponent for message generation
        num_messages: Number of messages to generate
        **kwargs: Additional arguments passed to evaluate_system
        
    Returns:
        Comprehensive metrics dictionary
    """
    message_set = generate_test_messages(vec_len, gf_exp, num_messages)
    return IdMetrics.evaluate_system(system, message_set, **kwargs)


def batch_evaluate_parameters(
    system_type: str,
    parameter_grid: Dict[str, List[Any]],
    vec_len: int,
    gf_exp: int,
    num_messages: int = 50,
    **eval_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a system type across multiple parameter combinations.
    
    Args:
        system_type: Type of system ("RSID", "RS2ID", etc.)
        parameter_grid: Dictionary of parameter names to lists of values
        vec_len: Vector length for message generation
        gf_exp: Galois field exponent
        num_messages: Number of test messages
        **eval_kwargs: Additional arguments for evaluation
        
    Returns:
        Dictionary mapping parameter combination strings to metrics
    """
    from .core import create_id_system
    from itertools import product
    
    results = {}
    message_set = generate_test_messages(vec_len, gf_exp, num_messages)
    
    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())
    
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))
        param_str = "_".join(f"{k}={v}" for k, v in params.items())
        
        try:
            system = create_id_system(system_type, params)
            results[param_str] = IdMetrics.evaluate_system(
                system, message_set, **eval_kwargs
            )
        except Exception as e:
            print(f"Failed to evaluate {param_str}: {e}")
            continue
    
    return results