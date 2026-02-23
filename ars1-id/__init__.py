"""
Intermediate Fusion Package
A multimodal deep learning model combining CLIP and ELECTRA for self-harm detection.
"""

__version__ = "1.0.0"
__author__ = "Elsa"

from .models import CLIPElectraFusion
from .data_loader import MultiModalDataset, collate_batch
from .train import train_one_epoch
from .evaluation import evaluate, plot_confusion_matrix, plot_training_history

__all__ = [
    "CLIPElectraFusion",
    "MultiModalDataset",
    "collate_batch",
    "train_one_epoch",
    "evaluate",
    "plot_confusion_matrix",
    "plot_training_history"
]
