#!/usr/bin/env python3
"""
Identification System Framework Package.

This package provides tools for creating, evaluating, and visualizing 
identification systems based on various coding schemes.
"""

from .core import (
    IdEncoder, IdDecoder, IdSystem,
    create_id_system, generate_numeric_messages, generate_string_messages
)
from .metrics import IdMetrics

__all__ = [
    'IdEncoder', 'IdDecoder', 'IdSystem',
    'HashTaggingEncoder', 'BitwiseCompareDecoder',
    'RandomProjectionEncoder', 'HammingDistanceDecoder',
    'create_id_system', 'generate_numeric_messages', 'generate_string_messages',
    'IdMetrics', 'IdVisualizer', 'InteractiveIdVisualizer'
]