"""
Processors module for RAG-Anything project.

This module contains specialized processors for different content types:
- ImageModalProcessor: For detailed visual analysis of images
- TableModalProcessor: For analyzing tabular data
- EquationModalProcessor: For mathematical equations and expressions
- GenericModalProcessor: For other content types
"""

from .base_processor import BaseModalProcessor
from .image_processor import ImageModalProcessor
from .table_processor import TableModalProcessor
from .equation_processor import EquationModalProcessor
from .generic_processor import GenericModalProcessor

__all__ = [
    "BaseModalProcessor",
    "ImageModalProcessor",
    "TableModalProcessor",
    "EquationModalProcessor",
    "GenericModalProcessor"
]