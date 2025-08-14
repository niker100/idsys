from typing import List, Any, Optional, Dict, Union, Generator
from ecidcodes.idcodes import IDCODES_U8, IDCODES_U16, IDCODES_U32, IDCODES_U64

# Type aliases for better code readability
Message = List[int]
Tag = Union[int, List[int]]
Parameters = Dict[str, Any]

def get_idcodes_instance(gf_exp: int):
    """
    Get the appropriate IDCODES instance based on Galois field exponent.
    
    The ecidcodes library provides different classes optimized for different
    Galois field sizes to balance performance and memory usage.
    
    Args:
        gf_exp: Galois field exponent (must be <= 64)
        
    Returns:
        Appropriate IDCODES instance
        
    Raises:
        ValueError: If gf_exp is too large or ecidcodes is not available
        ImportError: If ecidcodes library is not installed
    """
    if IDCODES_U8 is None:
        raise ImportError(
            "ecidcodes library is not available. Please install it or "
            "obtain the library from the official authors."
        )
    
    if gf_exp <= 8:
        return IDCODES_U8()
    elif gf_exp <= 16:
        return IDCODES_U16()
    elif gf_exp <= 32:
        return IDCODES_U32()
    elif gf_exp <= 64:
        return IDCODES_U64()
    else:
        raise ValueError(f"Unsupported GF exponent: {gf_exp}. Must be <= 64")