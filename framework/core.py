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
        message_length: (optional) length of the message in bytes
    """
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "nsym": 8,        # number of ECC symbols
            "code_length": 8, # tag length
            "message_length": 4 # message_length is optional
        }
        super().__init__(default_params)
        if parameters:
            self.set_parameters(parameters)
        self._init_rs()

    def _init_rs(self):
        # nsize will be set in set_parameters
        nsym = self.parameters.get("nsym", 8)
        message_length = self.parameters.get("message_length", 4)
        nsize = message_length + nsym
        self.parameters["nsize"] = nsize
        self.rs = reedsolo.RSCodec(
            nsym=nsym,
            nsize=nsize
        )

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        super().set_parameters(parameters)
        # Always update nsize
        nsym = self.parameters["nsym"]
        message_length = self.parameters["message_length"]
        self.parameters["nsize"] = message_length + nsym
        # Clamp code_length if needed
        if "code_length" in self.parameters and self.parameters["code_length"] > self.parameters["nsize"]:
            self.parameters["code_length"] = self.parameters["nsize"]
        self._init_rs()

    def encode(self, message: Any) -> Tuple[int, np.ndarray]:
        # Infer message_length from the message if not set
        if "message_length" not in self.parameters or self.parameters["message_length"] is None:
            self.parameters["message_length"] = self._infer_message_length(message)
            self.parameters["nsize"] = self.parameters["message_length"] + self.parameters["nsym"]
            self._init_rs()
        nsize = self.parameters["nsize"]
        nsym = self.parameters["nsym"]
        code_length = self.parameters["code_length"]
        k = nsize - nsym
        # Clamp code_length if needed
        if code_length > nsize:
            code_length = nsize
            self.parameters["code_length"] = code_length
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
        pi = random.randint(0, nsize - code_length)
        tag = codeword[pi:pi+code_length]
        return pi, tag


class PaperTaggingDecoder(IdDecoder):
    def __init__(self, encoder_parameters: Optional[Dict[str, Any]] = None, parameters: Optional[Dict[str, Any]] = None):
        default_params = {
            "nsym": 8,
            "code_length": 8,
            "message_length": 4  # default message length
        }
        super().__init__(default_params)
        if encoder_parameters:
            self.set_parameters(encoder_parameters)
        if parameters:
            self.set_parameters(parameters)
        self._init_rs()

    def _init_rs(self):
        nsym = self.parameters.get("nsym", 8)
        message_length = self.parameters.get("message_length", 4)
        nsize = message_length + nsym
        self.parameters["nsize"] = nsize
        self.rs = reedsolo.RSCodec(
            nsym=nsym,
            nsize=nsize
        )

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        if "message_length" not in parameters and "message_length" not in self.parameters:
            self.parameters["message_length"] = 4
        super().set_parameters(parameters)
        nsym = self.parameters["nsym"]
        message_length = self.parameters["message_length"]
        self.parameters["nsize"] = message_length + nsym
        if "code_length" in self.parameters and self.parameters["code_length"] > self.parameters["nsize"]:
            self.parameters["code_length"] = self.parameters["nsize"]
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
    # chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    chars = "AAAAAB"
    return [''.join(random.choice(chars) for _ in range(length)) for _ in range(count)]