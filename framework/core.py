#!/usr/bin/env python3
"""
Core module for identification systems.

This module provides the fundamental classes and functions for
implementing and evaluating identification systems.
"""

import numpy as np
import random
import reedsolo
from typing import List, Tuple, Any, Optional, Dict


class IdEncoder:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def encode(self, message: Any) -> Any:
        raise NotImplementedError()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)


class IdDecoder:
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        self.parameters = parameters or {}

    def decode(self, codeword: Any, message: Any) -> bool:
        raise NotImplementedError()

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)


class IdSystem:
    def __init__(self, encoder: IdEncoder, decoder: IdDecoder):
        self.encoder = encoder
        self.decoder = decoder

    def send(self, message: Any) -> Any:
        return self.encoder.encode(message)

    def receive(self, codeword: Any, message: Any) -> bool:
        return self.decoder.decode(codeword, message)


class PaperTaggingEncoder(IdEncoder):
    """
    Encoder using Reed-Solomon code for the paper's tagging scheme.
    Parameters:
        nsize: total codeword length (data+ecc)
        nsym: number of ECC symbols
        code_length: tag length (number of consecutive symbols to output)
    """
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "nsize": 32,      # total codeword length
            "nsym": 8,        # number of ECC symbols
            "code_length": 8  # tag length
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_rs()

    def _init_rs(self):
        self.rs = reedsolo.RSCodec(
            nsym=self.parameters["nsym"],
            nsize=self.parameters["nsize"]
        )

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_rs()

    def encode(self, message: Any) -> Tuple[int, np.ndarray]:
        nsize = self.parameters["nsize"]
        nsym = self.parameters["nsym"]
        code_length = self.parameters["code_length"]
        k = nsize - nsym
        # Convert message to bytes of length k
        if isinstance(message, int):
            msg_bytes = message.to_bytes(k, 'big', signed=False)
        elif isinstance(message, str):
            msg_bytes = message.encode('utf-8')[:k].ljust(k, b'\0')
        elif isinstance(message, bytes):
            msg_bytes = message[:k].ljust(k, b'\0')
        else:
            raise ValueError("Unsupported message type for RS encoding.")
        codeword = self.rs.encode(msg_bytes)
        codeword = np.frombuffer(codeword, dtype=np.uint8)
        if code_length > nsize:
            raise ValueError(f"code_length ({code_length}) must be <= nsize ({nsize})")
        pi = random.randint(0, nsize - code_length)
        tag = codeword[pi:pi+code_length]
        return pi, tag


class PaperTaggingDecoder(IdDecoder):
    def __init__(self, encoder_parameters: Optional[Dict[str, Any]] = None, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "nsize": 32,
            "nsym": 8,
            "code_length": 8
        }
        super().__init__(default_params)
        if encoder_parameters:
            self.set_parameters(encoder_parameters)
        if parameters:
            self.set_parameters(parameters)
        self._init_rs()

    def _init_rs(self):
        self.rs = reedsolo.RSCodec(
            nsym=self.parameters["nsym"],
            nsize=self.parameters["nsize"]
        )

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        self._init_rs()

    def _recompute_tag(self, message: Any, pi: int) -> np.ndarray:
        nsize = self.parameters["nsize"]
        nsym = self.parameters["nsym"]
        code_length = self.parameters["code_length"]
        k = nsize - nsym
        if isinstance(message, int):
            msg_bytes = message.to_bytes(k, 'big', signed=False)
        elif isinstance(message, str):
            msg_bytes = message.encode('utf-8')[:k].ljust(k, b'\0')
        elif isinstance(message, bytes):
            msg_bytes = message[:k].ljust(k, b'\0')
        else:
            raise ValueError("Unsupported message type for RS encoding.")
        codeword = self.rs.encode(msg_bytes)
        codeword = np.frombuffer(codeword, dtype=np.uint8)
        if not (0 <= pi <= nsize - code_length):
            raise ValueError(f"Index pi {pi} is out of bounds for nsize {nsize} and code_length {code_length}")
        tag = codeword[pi:pi+code_length]
        return tag

    def decode(self, codeword: Tuple[int, np.ndarray], message: Any) -> bool:
        if not isinstance(codeword, tuple) or len(codeword) != 2:
            return False
        pi, received_tag = codeword
        if not isinstance(received_tag, np.ndarray):
            return False
        try:
            recomputed_tag = self._recompute_tag(message, pi)
            if recomputed_tag.shape != received_tag.shape:
                return False
        except Exception:
            return False
        return np.array_equal(received_tag, recomputed_tag)


def create_id_system(system_type: str = "paper_tagging", parameters: Optional[Dict[str, Any]] = None) -> IdSystem:
    parameters = parameters or {}
    if system_type != "paper_tagging":
        raise ValueError("Only 'paper_tagging' system is supported.")
    encoder = PaperTaggingEncoder(parameters)
    decoder = PaperTaggingDecoder(encoder_parameters=parameters)
    return IdSystem(encoder, decoder)


def generate_numeric_messages(count: int, min_value: int = 0, max_value: int = 1000000) -> List[int]:
    return [random.randint(min_value, max_value) for _ in range(count)]


def generate_string_messages(count: int, length: int = 10) -> List[str]:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return [''.join(random.choice(chars) for _ in range(length)) for _ in range(count)]