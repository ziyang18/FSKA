"""Public FSKA model imports."""
from .text_model import OursClassifier as FSKAClassifier
from .vector_attention import FRSKAAttention as FSKAAttention

__all__ = ["FSKAClassifier", "FSKAAttention"]
