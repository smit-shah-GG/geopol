"""
TKG prediction models.
"""

from .regcn_cpu import REGCN, ConvTransEDecoder

__all__ = ["REGCN", "ConvTransEDecoder"]
