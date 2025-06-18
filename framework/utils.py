"""
Utility functions for evaluating identification systems.
"""

from typing import List, Dict, Any
from .core import IdSystem, generate_test_messages
from .metrics import IdMetrics


def batch_evaluate_parameters(
    system: IdSystem,
    parameter_grid: Dict[str, List[Any]],
    vec_len: int,
    num_messages: int = 100,
    **eval_kwargs
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate a system type across multiple parameter combinations.
    
    Args:
        system: The identification system to evaluate
        parameter_grid: Dictionary of parameter names to lists of values
        vec_len: Vector length for message generation
        num_messages: Number of test messages
        **eval_kwargs: Additional arguments for evaluation
        
    Returns:
        Dictionary mapping parameter combination strings to metrics dictionaries
    """
    from itertools import product
    
    results = {}
    
    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())
    
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))
        param_str = "_".join(f"{k}={v}" for k, v in params.items())    

        # Use the gf_exp from params, if not provided, fallback to system's encoder parameters
        current_gf_exp = params.get('gf_exp', system.encoder.parameters.get('gf_exp'))
        if current_gf_exp is None:
            raise ValueError(f"Parameter 'gf_exp' is neither in the provided parameters nor in the system's encoder parameters.")
        
        # Generate messages with the correct gf_exp for this parameter combination
        message_set = generate_test_messages(vec_len, current_gf_exp, num_messages)
        
        # Update system parameters
        system.encoder.set_parameters(params)
        system.verifier.set_parameters(params)


        results[param_str] = IdMetrics.evaluate_system(
            system, message_set, **eval_kwargs
        )

    return results