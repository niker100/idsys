"""
Identification System Framework

A modular framework for evaluating different metrics of identification systems.
"""

from .core import Sender, Receiver, IdentificationSystem
from .encoders import Encoder, TaggingEncoder
from .metrics import Metrics, Evaluator

__all__ = [
    'Sender',
    'Receiver',
    'IdentificationSystem',
    'Encoder',
    'TaggingEncoder',
    'Metrics',
    'Evaluator'
]