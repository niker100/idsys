#!/usr/bin/env python3
"""
Core module for identification systems using the idcodes library.

This module provides the fundamental classes and functions for
implementing and evaluating identification systems with multiple coding schemes.
"""
from typing import List, Any, Optional, Dict
from idcodes.idcodes import IDCODES_U8, IDCODES_U16, IDCODES_U32, IDCODES_U64


class IdEncoder:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def encode(self, message: Any) -> Any:
        raise NotImplementedError("Subclasses must implement encode method")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)


class IdVerifier:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def verify(self, codeword: Any, message: Any) -> bool:
        raise NotImplementedError("Subclasses must implement verify method")

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)


class IdSystem:
    def __init__(self, encoder: IdEncoder, verifier: IdVerifier):
        self.encoder = encoder
        self.verifier = verifier

    def send(self, message: Any) -> Any:
        return self.encoder.encode(message)

    def receive(self, codeword: Any, message: Any) -> bool:
        return self.verifier.verify(codeword, message)


def _get_idcodes_instance(gf_exp: int):
    """Get the appropriate IDCODES instance based on Galois field exponent."""
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


class RSIDEncoder(IdEncoder):
    """Reed-Solomon Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": 2
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        self.idcodes.generate_gf_outer(self.gf_exp)
        self.exp_arr = self.idcodes.get_exp_arr()
        self.log_arr = self.idcodes.get_log_arr()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        tag_pos = self.parameters["tag_pos"]
        result = self.idcodes.rsid(message, tag_pos, self.exp_arr, self.log_arr, self.gf_exp)
        return result


class RSIDVerifier(IdVerifier):
    """Reed-Solomon Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RSIDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class RS2IDEncoder(IdEncoder):
    """Concatenated Reed-Solomon Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": 2,
            "tag_pos_in": 2
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        self.idcodes.generate_gf_outer(self.gf_exp)
        self.idcodes.generate_gf_inner(self.gf_exp)
        
        self.exp_arr = self.idcodes.get_exp_arr()
        self.log_arr = self.idcodes.get_log_arr()
        self.exp_arr_in = self.idcodes.get_exp_arr_in()
        self.log_arr_in = self.idcodes.get_log_arr_in()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        tag_pos = self.parameters["tag_pos"]
        tag_pos_in = self.parameters["tag_pos_in"]
        
        result = self.idcodes.rs2id(
            message, tag_pos, tag_pos_in, 
            self.exp_arr, self.log_arr, 
            self.exp_arr_in, self.log_arr_in, 
            self.gf_exp
        )
        return result


class RS2IDVerifier(IdVerifier):
    """Concatenated Reed-Solomon Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RS2IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class RMIDEncoder(IdEncoder):
    """Reed-Muller Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8,
            "tag_pos": 2,
            "rm_order": 1
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
        self.idcodes.generate_gf_outer(self.gf_exp)
        self.exp_arr = self.idcodes.get_exp_arr()
        self.log_arr = self.idcodes.get_log_arr()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        tag_pos = self.parameters["tag_pos"]
        rm_order = self.parameters["rm_order"]
        
        result = self.idcodes.rmid(message, tag_pos, rm_order, self.exp_arr, self.log_arr, self.gf_exp)
        return result


class RMIDVerifier(IdVerifier):
    """Reed-Muller Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = RMIDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class SHA1IDEncoder(IdEncoder):
    """SHA1-based Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        result = self.idcodes.sha1id(message, self.gf_exp)
        return result


class SHA1IDVerifier(IdVerifier):
    """SHA1-based Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = SHA1IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


class SHA256IDEncoder(IdEncoder):
    """SHA256-based Identification encoder using idcodes library."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "gf_exp": 8
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_idcodes()
    
    def _init_idcodes(self):
        self.gf_exp = self.parameters["gf_exp"]
        self.idcodes = _get_idcodes_instance(self.gf_exp)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_idcodes()
    
    def encode(self, message: List[int]) -> int:
        result = self.idcodes.sha256id(message, self.gf_exp)
        return result


class SHA256IDVerifier(IdVerifier):
    """SHA256-based Identification verifier."""
    
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__(parameters)
        self.encoder = SHA256IDEncoder(parameters)
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self.encoder.set_parameters(parameters)
    
    def verify(self, codeword: int, message: List[int]) -> bool:
        recomputed_tag = self.encoder.encode(message)
        return recomputed_tag == codeword


def create_id_system(system_type: str = "RSID", parameters: Optional[Dict[str, Any]] = None) -> IdSystem:
    """
    Create an identification system with the specified encoder/verifier type.
    
    Args:
        system_type: One of "RSID", "RS2ID", "RMID", "SHA1ID", "SHA256ID"
        parameters: System parameters (gf_exp, tag_pos, tag_pos_in, rm_order)
    """
    parameters = parameters or {}
    
    systems = {
        "RSID": (RSIDEncoder, RSIDVerifier),
        "RS2ID": (RS2IDEncoder, RS2IDVerifier),
        "RMID": (RMIDEncoder, RMIDVerifier),
        "SHA1ID": (SHA1IDEncoder, SHA1IDVerifier),
        "SHA256ID": (SHA256IDEncoder, SHA256IDVerifier)
    }
    
    if system_type not in systems:
        raise ValueError(f"Unsupported system type: {system_type}. "
                        f"Supported types: {list(systems.keys())}")
    
    encoder_class, verifier_class = systems[system_type]
    encoder = encoder_class(parameters)
    verifier = verifier_class(parameters)
    
    return IdSystem(encoder, verifier)


def generate_test_messages(vec_len: int, gf_exp: int, count: int = 1) -> List[List[int]]:
    """
    Generate test messages using the same approach as Pybenchmark.
    
    Args:
        vec_len: Vector length in elements
        gf_exp: Galois field exponent
        count: Number of messages to generate
        
    Returns:
        List of integer lists representing messages
    """
    # Adjust vector length based on GF exponent (like Pybenchmark)
    if gf_exp >= 33:
        vec_len_ = vec_len // 8
    elif gf_exp >= 17:
        vec_len_ = vec_len // 4
    elif gf_exp >= 9:
        vec_len_ = vec_len // 2
    else:
        vec_len_ = vec_len
    
    # Get appropriate idcodes instance
    Id = _get_idcodes_instance(gf_exp)
    
    messages = []
    for _ in range(count):
        message = Id.generate_string_sequence(vec_len)
        messages.append(message)
    
    return messages