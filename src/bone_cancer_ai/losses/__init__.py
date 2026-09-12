"""Task-aware losses for HOSP-BoneNet."""

from .classification import masked_binary_cross_entropy, masked_cross_entropy
from .concepts import masked_concept_loss
from .grounding import spatial_grounding_loss
from .hierarchy import hierarchy_consistency_loss
from .segmentation import SegmentationLoss, dice_bce_segmentation_loss, segmentation_loss
from .total import HOSPLoss, LossWeights

__all__ = [
    "HOSPLoss",
    "LossWeights",
    "masked_binary_cross_entropy",
    "masked_cross_entropy",
    "masked_concept_loss",
    "spatial_grounding_loss",
    "hierarchy_consistency_loss",
    "SegmentationLoss",
    "dice_bce_segmentation_loss",
    "segmentation_loss",
]
