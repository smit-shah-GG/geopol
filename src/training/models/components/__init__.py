"""TiRGN model components.

Submodules implementing the global history encoder and Time-ConvTransE
decoder that differentiate TiRGN from RE-GCN.
"""

from .global_history import GlobalHistoryEncoder, build_history_vocabulary
from .time_conv_transe import TimeConvTransEDecoder

__all__ = [
    "GlobalHistoryEncoder",
    "TimeConvTransEDecoder",
    "build_history_vocabulary",
]
