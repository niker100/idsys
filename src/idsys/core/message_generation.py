"""
Module for generating test messages for identification systems.
"""
import numpy as np
from .common import *
from time import time

def generate_test_messages(
    vec_len: int, 
    gf_exp: int, 
    count: int = 1
) -> List[Message]:
    """
    Generate random test messages for identification system evaluation.
    
    This function generates test messages using the same approach as the benchmark
    suite, with automatic vector length adjustment based on Galois field parameters.
    
    Args:
        vec_len: Target vector length in elements
        gf_exp: Galois field exponent (affects actual vector length)
        count: Number of messages to generate
        
    Returns:
        List of messages, where each message is a list of integers
        
    Raises:
        ImportError: If ecidcodes library is not available
        ValueError: If parameters are invalid
        
    Example:
        >>> messages = generate_test_messages(vec_len=16, gf_exp=8, count=5)
        >>> print(f"Generated {len(messages)} messages")
        >>> print(f"First message: {messages[0]}")
    
    Note:
        Vector length is automatically adjusted based on GF exponent to maintain
        consistent memory usage and performance characteristics across different
        field sizes.
    """
    if count <= 0:
        raise ValueError("Count must be positive")
    if vec_len <= 0:
        raise ValueError("Vector length must be positive")
    if gf_exp <= 0:
        raise ValueError("GF exponent must be positive")
    
    # Adjust vector length based on GF exponent for performance optimization
    if gf_exp >= 33:
        vec_len_adjusted = vec_len // 8
    elif gf_exp >= 17:
        vec_len_adjusted = vec_len // 4
    elif gf_exp >= 9:
        vec_len_adjusted = vec_len // 2
    else:
        vec_len_adjusted = vec_len
    
    # Ensure minimum vector length
    vec_len_adjusted = max(1, vec_len_adjusted)
    
    # Get appropriate idcodes instance
    idcodes_instance = get_idcodes_instance(gf_exp)
    
    messages = []
    for _ in range(count):
        message = idcodes_instance.generate_string_sequence(vec_len_adjusted)
        messages.append(message)
    
    return messages


def generate_structured_messages(
    vec_len: int,
    pattern_type: str,
    gf_exp: int,
    target_count: int = 5000,
    generate_first: bool = False,
    seed: Optional[int] = None,
    worker_offset: int = 0
) -> Generator[Message, None, None]:
    """
    Generator for creating messages with specific structural patterns.
    
    This generator produces messages with controlled patterns, useful for testing
    identification systems under different data conditions. It supports various
    pattern types and is designed to work efficiently with multiprocessing.
    
    Args:
        vec_len: Vector length in elements
        pattern_type: Pattern type to generate. Options:
                     - "random": Random messages (using idcodes)
                     - "incremental": Messages with incremental last element
                     - "repeated_patterns": Messages with repeating byte patterns
                     - "sparse": Messages with mostly zeros and few non-zero elements
                     - "low_entropy": Messages using limited alphabet
                     - "only_two": Messages with only two distinct values
        gf_exp: Galois field exponent
        target_count: Number of messages to generate
        generate_first: If True, only yield the first message and return
        seed: Random seed for reproducibility (optional)
        worker_offset: Offset for multiprocessing to avoid duplicate sequences
        
    Yields:
        Messages matching the specified pattern
        
    Raises:
        ValueError: If pattern_type is not supported or parameters are invalid
        
    Example:
        >>> # Generate sparse messages
        >>> messages = list(generate_structured_messages(
        ...     vec_len=16, 
        ...     pattern_type="sparse", 
        ...     gf_exp=8, 
        ...     target_count=10
        ... ))
        >>> 
        >>> # Generate low entropy messages with seed
        >>> low_entropy_gen = generate_structured_messages(
        ...     vec_len=32,
        ...     pattern_type="low_entropy",
        ...     gf_exp=8,
        ...     target_count=100,
        ...     seed=42
        ... )
    
    Note:
        This function uses a static variable to store the first message across
        calls, enabling consistent collision testing. The worker_offset parameter
        ensures different processes generate distinct sequences while maintaining
        reproducibility.
    """
    if target_count <= 0:
        raise ValueError("Target count must be positive")
    if vec_len <= 0:
        raise ValueError("Vector length must be positive")
    if gf_exp <= 0:
        raise ValueError("GF exponent must be positive")
    
    supported_patterns = {
        "random", "incremental", "repeated_patterns", 
        "sparse", "low_entropy", "only_two"
    }
    if pattern_type not in supported_patterns:
        raise ValueError(
            f"Unsupported pattern type: {pattern_type}. "
            f"Supported patterns: {supported_patterns}"
        )
    
    # Static variable to store the first message across calls
    if not hasattr(generate_structured_messages, "_first_message"):
        generate_structured_messages._first_message = None
    
    # Set up random seed for reproducibility
    if seed is not None:
        base_seed = seed
    else:
        base_seed = int(time())
    
    # Create process-specific random state
    process_seed = base_seed + worker_offset
    process_random = np.random.RandomState(process_seed)

    # Get idcodes instance and adjust vector length
    idcodes_instance = get_idcodes_instance(gf_exp)
    if gf_exp >= 33:
        vec_len_adjusted = vec_len // 8
    elif gf_exp >= 17:
        vec_len_adjusted = vec_len // 4
    elif gf_exp >= 9:
        vec_len_adjusted = vec_len // 2
    else:
        vec_len_adjusted = vec_len
    
    vec_len_adjusted = max(1, vec_len_adjusted)

    def _generate_pattern_message(attempt: int) -> Message:
        """Generate a single message for the given attempt number."""
        # Use process-specific attempt counter to avoid overlaps
        effective_attempt = attempt + (worker_offset * 1000)
        
        if pattern_type == "random":
            return idcodes_instance.generate_string_sequence(vec_len_adjusted)
            
        elif pattern_type == "incremental":
            return [0] * (vec_len - 1) + [effective_attempt % (2 ** 8)]
            
        elif pattern_type == "repeated_patterns":
            # Different base patterns for different workers
            base_patterns = [
                [255, 0],
                [170, 187],          # 0xAA, 0xBB
                [85, 170],           # 0x55, 0xAA
                [1, 2, 3, 4],
                [15, 240, 15, 240],  # 0x0F, 0xF0
                [10, 20, 30, 40]
            ]
            base_pattern = base_patterns[worker_offset % len(base_patterns)]
            
            # Modify pattern with attempt number for uniqueness
            pattern = [
                (p + effective_attempt) % (2**gf_exp) 
                for p in base_pattern
            ]
            
            # Tile pattern to fill message vector
            num_repeats = (vec_len + len(pattern) - 1) // len(pattern)
            return (pattern * num_repeats)[:vec_len]
            
        elif pattern_type == "sparse":
            msg = [0] * vec_len
            num_nonzero = 1 + (effective_attempt % 3)
            
            # Use different prime offsets for each worker
            prime_offsets = [7, 11, 13, 17, 19, 23, 29, 31]
            prime_offset = prime_offsets[worker_offset % len(prime_offsets)]
            
            positions = [
                (effective_attempt + j * prime_offset) % vec_len 
                for j in range(num_nonzero)
            ]
            for pos in positions:
                msg[pos] = 1 + (effective_attempt + pos) % (2 ** gf_exp - 1)
            return msg
            
        elif pattern_type == "low_entropy":
            # Use limited alphabet for low entropy
            alphabet = [0, 1, 2, 3]
            return process_random.choice(alphabet, size=vec_len).tolist()
        
        elif pattern_type == "only_two":
            # Alternate between two patterns
            if effective_attempt % 2 == 0:
                return [0] * vec_len
            else:
                return [1] * vec_len
        
        else:
            # Should never reach here due to earlier validation
            raise ValueError(f"Unsupported pattern type: {pattern_type}")

    # Handle first message generation and storage
    if generate_first or generate_structured_messages._first_message is None:
        first_message = (
            _generate_pattern_message(0) 
            if worker_offset == 0 
            else generate_structured_messages._first_message
        )
        generate_structured_messages._first_message = first_message
        
        if generate_first:
            yield first_message
            return

    first_message = generate_structured_messages._first_message

    # Generate unique messages (avoiding the first message)
    count = 0
    attempts = 1  # Start from 1 to avoid duplicating first message
    max_attempts = target_count * 10  # Prevent infinite loops

    while count < target_count and attempts < max_attempts:
        msg = _generate_pattern_message(attempts)
        attempts += 1
        
        # Only yield if different from first message
        if msg != first_message:
            yield msg
            count += 1

    if count < target_count:
        print(f"Warning: Could only generate {count}/{target_count} unique messages")