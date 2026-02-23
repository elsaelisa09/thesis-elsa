"""
Main training script for the Intermediate Fusion model.
"""

import os
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModel
import wandb

from data_loader import MultiModalDataset, collate_batch

# ============= MODEL SELECTION =============
# Uncomment the model you want to use:
from models import CLIPElectraFusion, EarlyStopping  # Model 1 (Transformer Fusion)
# from model2 import CLIPElectraFusion, EarlyStopping  # Model 2 (Alternative)
# from model3 import CLIPElectraFusion, EarlyStopping  # Model 3
MODEL_NAME = 'model1'  # Change this to match your model
# ===========================================

from train import train_one_epoch, setup_optimizer, setup_scheduler
from evaluation import (
    evaluate, 
    plot_confusion_matrix, 
    plot_training_history,
    print_classification_report,
    analyze_model_parameters
)


class Config:
    """Configuration class for training parameters."""
    
    # Paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULTS_DIR = os.path.join(CURRENT_DIR, 'results')
    DATASET_DIR = os.path.join(os.path.dirname(CURRENT_DIR), 'dataset')
    IMAGES_DIR = os.path.join(DATASET_DIR, 'fix-review-manual-label')
    LABELS_CSV = os.path.join(DATASET_DIR, 'fix-review-manual-label.csv')
    
    # Model Configuration
    MODEL_NAME = MODEL_NAME  # From import section above
    
    # Training Parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16
    EPOCHS = 12
    LR_PRETRAIN = 1e-5
    LR_HEAD = 5e-4
    MAX_LEN = 128
    NUM_CLASSES = 2
    IMAGE_SIZE = 224
    SEED = 42
    NUM_WORKERS = 2
    PATIENCE = 3
    
    # Model Architecture Parameters
    FUSION_IMG_DIM = 512
    FUSION_TEXT_DIM = 256


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data(config):

    assert os.path.exists(config.LABELS_CSV), f'CSV not found: {config.LABELS_CSV}'
    assert os.path.isdir(config.IMAGES_DIR), f'Images folder not found: {config.IMAGES_DIR}'
    
    labels_df = pd.read_csv(config.LABELS_CSV)
    label_mapping = {'NON-SELF-HARM': 0, 'SELF-HARM': 1}
    
    if 'Label Akhir' not in labels_df.columns or 'filename' not in labels_df.columns:
        raise ValueError('CSV must have columns: Label Akhir, filename, and Teks Terlihat (optional)')
    
    labels_df = labels_df[labels_df['Label Akhir'].isin(label_mapping.keys())].copy()
    labels_df['Label'] = labels_df['Label Akhir'].map(label_mapping)
    
    print(f'\nTotal samples after filtering: {len(labels_df)}')
    print(f'Overall label distribution:')
    for label, count in labels_df['Label'].value_counts().sort_index().items():
        label_name = 'NON-SELF-HARM' if label == 0 else 'SELF-HARM'
        print(f'  {label} ({label_name}): {count} ({count/len(labels_df)*100:.1f}%)')
    
    train_df, val_df = train_test_split(
        labels_df, 
        test_size=0.2, 
        random_state=config.SEED, 
        stratify=labels_df['Label']
    )
    
    print(f'\nTrain set: {len(train_df)} samples')
    for label, count in train_df['Label'].value_counts().sort_index().items():
        label_name = 'NON-SELF-HARM' if label == 0 else 'SELF-HARM'
        print(f'  {label} ({label_name}): {count} ({count/len(train_df)*100:.1f}%)')
    
    print(f'\nValidation set: {len(val_df)} samples')
    for label, count in val_df['Label'].value_counts().sort_index().items():
        label_name = 'NON-SELF-HARM' if label == 0 else 'SELF-HARM'
        print(f'  {label} ({label_name}): {count} ({count/len(val_df)*100:.1f}%)')
    
    clip_model_name = 'openai/clip-vit-base-patch32'
    electra_model_name = 'sentinet/suicidality'
    
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
    electra_tokenizer = AutoTokenizer.from_pretrained(electra_model_name)
    
    train_ds = MultiModalDataset(
        train_df, config.IMAGES_DIR, electra_tokenizer, 
        clip_processor, max_len=config.MAX_LEN, is_train=True
    )
    val_ds = MultiModalDataset(
        val_df, config.IMAGES_DIR, electra_tokenizer, 
        clip_processor, max_len=config.MAX_LEN, is_train=False
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_batch, 
        pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, 
        num_workers=config.NUM_WORKERS, collate_fn=collate_batch, 
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, clip_processor, electra_tokenizer


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Intermediate Fusion Model')
    parser.add_argument('--images_dir', type=str, default=None, 
                        help='Path to images directory')
    parser.add_argument('--labels_csv', type=str, default=None, 
                        help='Path to labels CSV file')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None, 
                        help='Number of training epochs')
    parser.add_argument('--lr_pretrain', type=float, default=None, 
                        help='Learning rate for pretrained layers')
    parser.add_argument('--lr_head', type=float, default=None, 
                        help='Learning rate for head layers')
    parser.add_argument('--wandb_project', type=str, default='intermediate-fusion', 
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, 
                        help='W&B run name')
    parser.add_argument('--no_wandb', action='store_true', 
                        help='Disable W&B logging')
    parser.add_argument('--notes', type=str, default=None, 
                        help='percobaan1')
    parser.add_argument('--tags', type=str, nargs='+', default=None, 
                        help='thesis')
    args = parser.parse_args()
    
    config = Config()
    if args.images_dir:
        config.IMAGES_DIR = args.images_dir
    if args.labels_csv:
        config.LABELS_CSV = args.labels_csv
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.lr_pretrain:
        config.LR_PRETRAIN = args.lr_pretrain
    if args.lr_head:
        config.LR_HEAD = args.lr_head
    
    set_seed(config.SEED)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    
    # Save training notes to local file
    if args.notes:
        notes_file = os.path.join(config.RESULTS_DIR, 'training_notes.txt')
        with open(notes_file, 'a', encoding='utf-8') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"\n{'='*80}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Run Name: {args.wandb_name or 'default'}\n")
            f.write(f"Notes: {args.notes}\n")
            if args.tags:
                f.write(f"Tags: {', '.join(args.tags)}\n")
            f.write(f"{'='*80}\n")
        print(f'Notes disimpan ke: {notes_file}')
    
    # Initialize Weights & Biases
    use_wandb = not args.no_wandb
    if use_wandb:
        # Auto-include model name in wandb run name if not provided
        wandb_run_name = args.wandb_name if args.wandb_name else 'p1-ars1-id'
        
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            notes=args.notes,
            tags=args.tags,
            config={
                'model_name': config.MODEL_NAME,
                'batch_size': config.BATCH_SIZE,
                'epochs': config.EPOCHS,
                'lr_pretrain': config.LR_PRETRAIN,
                'lr_head': config.LR_HEAD,
                'max_len': config.MAX_LEN,
                'num_classes': config.NUM_CLASSES,
                'image_size': config.IMAGE_SIZE,
                'seed': config.SEED,
                'patience': config.PATIENCE,
                'fusion_img_dim': config.FUSION_IMG_DIM,
                'fusion_text_dim': config.FUSION_TEXT_DIM,
            }
        )
        print(f'W&B initialized: {wandb.run.name}')
        if args.notes:
            print(f'Notes: {args.notes}')
        if args.tags:
            print(f'Tags: {", ".join(args.tags)}')
    
    print(f'MODEL: {config.MODEL_NAME}')
    print(f'Device: {config.DEVICE}')
    print(f'Results directory: {config.RESULTS_DIR}')
    print(f'CSV path: {config.LABELS_CSV}')
    print(f'Images directory: {config.IMAGES_DIR}')
    
    train_loader, val_loader, clip_processor, electra_tokenizer = load_data(config)
    
    print('\nLoading pretrained models...')
    clip_model_name = 'openai/clip-vit-base-patch32'
    electra_model_name = 'sentinet/suicidality'
    
    clip_model = CLIPModel.from_pretrained(clip_model_name)
    electra_model = AutoModel.from_pretrained(electra_model_name)
    
    print('Initializing fusion model...')
    model = CLIPElectraFusion(
        clip_model=clip_model,
        electra_model=electra_model,
        fusion_img_dim=config.FUSION_IMG_DIM,
        fusion_text_dim=config.FUSION_TEXT_DIM,
        num_classes=config.NUM_CLASSES,
        freeze_encoders=True
    )
    model = model.to(config.DEVICE)
    
    analyze_model_parameters(model)
    
    optimizer = setup_optimizer(model, config.LR_PRETRAIN, config.LR_HEAD)
    scheduler = setup_scheduler(optimizer)
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=config.PATIENCE, mode='max')
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'lrs': []
    }
    
    # Output filenames with model name
    best_model_path = os.path.join(config.RESULTS_DIR, f'bestmodel_{config.MODEL_NAME}.pth')
    best_f1 = -1
    
    print('\nStarting training...\n')
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f'Epoch {epoch}/{config.EPOCHS}')
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, config.DEVICE
        )
        
        val_loss, val_acc, p, r, f1, cm, val_preds, val_trues = evaluate(
            model, val_loader, criterion, config.DEVICE
        )
        
        scheduler.step(val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        history['lrs'].append(current_lr)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}')
        print(f'Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}')
        print(f'Learning Rate: {current_lr:.2e}')
        print(f'Confusion Matrix:\n{cm}\n')
        
        # Log metrics to W&B
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'val/precision': p,
                'val/recall': r,
                'val/f1': f1,
                'learning_rate': current_lr,
            })
        
        improved = early_stopper.step(f1)
        
        if improved:
            print(f'Best model saved (F1 improved to {f1:.4f})')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': f1,
                'history': history
            }, best_model_path)
            best_f1 = f1
            
            # Log best model to W&B
            if use_wandb:
                wandb.run.summary['best_f1'] = f1
                wandb.run.summary['best_epoch'] = epoch
        else:
            print(f'No improvement ({early_stopper.num_bad}/{config.PATIENCE})')
        
        if early_stopper.should_stop:
            print('\nEarly stopping triggered!')
            break
        
        print()
    
    print(f'\nTraining completed. Best F1: {best_f1:.4f}')
    print(f'Best model saved to: {best_model_path}')
    
    print('\nLoading best model for final evaluation...')
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, val_acc, p, r, f1, cm, val_preds, val_trues = evaluate(
        model, val_loader, criterion, config.DEVICE
    )
    
    print('\nFinal Validation Results:')
    print(f'Accuracy:  {val_acc:.4f}')
    print(f'Precision: {p:.4f}')
    print(f'Recall:    {r:.4f}')
    print(f'F1 Score:  {f1:.4f}')
    
    print_classification_report(val_trues, val_preds)
    
    # Save plots with model name
    cm_path = os.path.join(config.RESULTS_DIR, f'confusion_matrix_{config.MODEL_NAME}.png')
    plot_confusion_matrix(cm, save_path=cm_path)
    
    history_path = os.path.join(config.RESULTS_DIR, f'training_history_{config.MODEL_NAME}.png')
    plot_training_history(history, save_path=history_path)
    
    print(f'\nPlots saved to {config.RESULTS_DIR}')
    
    # Log final artifacts to W&B
    if use_wandb:
        wandb.log({
            'confusion_matrix': wandb.Image(cm_path),
            'training_history': wandb.Image(history_path)
        })
        # Save model as artifact
        artifact = wandb.Artifact('model', type='model')
        artifact.add_file(best_model_path)
        wandb.log_artifact(artifact)
        wandb.finish()
        print('W&B artifacts logged and run finished')


if __name__ == '__main__':
    main()
