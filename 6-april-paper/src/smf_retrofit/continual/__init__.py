"""Sparse update selection and masking."""

from smf_retrofit.continual.gradient_masking import GradientMaskManager
from smf_retrofit.continual.selection import KLSlotSelector, TFIDFSlotSelector

__all__ = ["GradientMaskManager", "KLSlotSelector", "TFIDFSlotSelector"]
