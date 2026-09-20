"""Question-conditioned, extractive meeting-note compression.

ML dependencies are optional and imported only by model/training commands.
"""

from .spans import PruneResult, Span, assemble, segment

__all__ = ["PruneResult", "Span", "assemble", "segment"]
