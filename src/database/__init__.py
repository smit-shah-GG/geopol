"""
Database module for GDELT event storage and retrieval.
"""

from .models import Event
from .storage import EventStorage

__all__ = [
    'Event',
    'EventStorage',
]