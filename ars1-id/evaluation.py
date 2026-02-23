"""
Evaluation and metrics module.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    confusion_matrix,
    classification_report
)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds, trues = [], []
    
    for batch in tqdm(loader, desc="Evaluating"):
        pixel = batch['pixel_values'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits, _, _ = model(pixel, ids, mask)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds.extend(logits.argmax(-1).detach().cpu().tolist())
        trues.extend(labels.detach().cpu().tolist())
    
    acc = accuracy_score(trues, preds)
    p, r, f1, _ = precision_recall_fscore_support(trues, preds, average='binary', zero_division=0)
    cm = confusion_matrix(trues, preds)
    
    return total_loss / len(loader), acc, p, r, f1, cm, preds, trues


def plot_confusion_matrix(cm, class_names=['NON-SELF-HARM', 'SELF-HARM'], save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_training_history(history, save_path=None):
    """
    Plot training history including loss and accuracy curves.
    
    Args:
        history: Dictionary containing training history
        save_path: Path to save the plot (optional)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Accuracy')
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def print_classification_report(trues, preds, class_names=['NON-SELF-HARM', 'SELF-HARM']):
    """
    Print detailed classification report.
    
    Args:
        trues: True labels
        preds: Predicted labels
        class_names: List of class names
    """
    print("\nClassification Report:")
    print("=" * 60)
    print(classification_report(trues, preds, target_names=class_names))


def analyze_model_parameters(model):

    print("MODEL PARAMETER ANALYSIS")

    def count(p_list):
        return sum(p.numel() for p in p_list)

    clip_vision_params = list(model.clip.vision_model.parameters())
    print(f"CLIP Vision Encoder: {count(clip_vision_params):,} params | "
          f"Trainable: {count([p for p in clip_vision_params if p.requires_grad]):,}")

    clip_text_params = list(model.clip.text_model.parameters())
    print(f"CLIP Text Encoder:   {count(clip_text_params):,} params | "
          f"Trainable: {count([p for p in clip_text_params if p.requires_grad]):,}")

    electra_params = list(model.electra.parameters())
    print(f"ELECTRA Encoder:     {count(electra_params):,} params | "
          f"Trainable: {count([p for p in electra_params if p.requires_grad]):,}")

    # CLIP tidak punya projection lagi (langsung 512 dim)
    # Hanya text projection
    proj_text_params = list(model.project_text.parameters())

    projection_total = count(proj_text_params)
    projection_trainable = count([p for p in proj_text_params if p.requires_grad])
    print(f"Text Projection:     {projection_total:,} params | Trainable: {projection_trainable:,}")

    # Check if model has Transformer fusion (only in models.py, not in model2.py)
    if hasattr(model, 'fusion_transformer'):
        fusion_transformer_params = list(model.fusion_transformer.parameters())
        print(f"Fusion Transformer:  {count(fusion_transformer_params):,} params | "
              f"Trainable: {count([p for p in fusion_transformer_params if p.requires_grad]):,}")
    else:
        print(f"Fusion Transformer:  0 params | Trainable: 0 (NO TRANSFORMER - Late Fusion)")

    # Check if model has positional embedding (only in models.py, not in model2.py)
    if hasattr(model, 'pos_embedding'):
        pos_emb_params = model.pos_embedding.numel()
        print(f"Positional Embed:    {pos_emb_params:,} params | Trainable: {pos_emb_params:,}")

    classifier_params = list(model.classifier.parameters())
    print(f"Classifier:          {count(classifier_params):,} params | "
          f"Trainable: {count([p for p in classifier_params if p.requires_grad]):,}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total Parametes:        {total_params:,}")
    print(f"Trainable Parameters:    {trainable_params:,}")
    print(f"Frozen Parameters:       {total_params - trainable_params:,}")
    print(f"Trainable %:         {100 * trainable_params / total_params:.3f}%")

    return {
        "total": total_params,
        "trainable": trainable_params,
        "frozen": total_params - trainable_params
    }
