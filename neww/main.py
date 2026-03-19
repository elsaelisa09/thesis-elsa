import os #path file
import random #seed random python
import argparse #Parsing argumen CLI 
import numpy as np #operasi array
import pandas as pd #operasi data frame (csv)
import torch #pytorch
import torch.nn as nn #modul neural network pytorch
from torch.utils.data import DataLoader #modul untuk memuat data dalam bentuk batch
from sklearn.model_selection import train_test_split #untuk membagi data menjadi training dan testing
from transformers  import CLIPImageProcessor, CLIPVisionModelWithProjection, AutoTokenizer, AutoModel #modul untuk menggunakan model CLIP dan tokenizer
import wandb #untuk logging dan visualisasi hasil training

from data_loader import MultimodalDataset #dataset untuk data multimodal (text + image)
from models import CLIPElectraFusion, EarlyStopping #model untuk menggabungkan fitur dari CLIP dan ELECTRA
MODEL_NAME = 'clipvision'

from train import train_one_epoch, setup_optimizer, setup_scheduler, FocalLoss 
from evaluation import (
    analyze_model_parameters,
    evaluate,
    plot_confusion_matrix,
    plot_training_history,
    print_classification_report
)

class Config:
    #path file
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(CURRENT_DIR, 'result')
    DATASET_DIR = os.path.join(CURRENT_DIR, 'dataset')
    IMAGES_DIR = os.path.join(DATASET_DIR, 'fix-review-manual-label')
    LABELS_CSV = os.path.join(DATASET_DIR, 'fix-review-manual-label.csv')

    MODEL_NAME = MODEL_NAME

    #training parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 16
    EPOCHS = 12
    LEARNING_RATE = 1e-5
    MAX_LEN = 128
    NUM_CLASSES = 2
    IMAGE_SIZE = 224
    SEED = 42
    NUM_WORKERS = 4
    PATIENCE = 3
    FUSION_TEXT_DIM = 256

    #pretrained model names 
    CLIP_MODEL_NAME = 'openai/clip-vit-base-patch32'
    ELECTRA_MODEL_NAME = 'sentinet/suicidality'

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(config):

    assert os.path.exists(config.LABELS_CSV), f'CSV not found: {config.LABELS_CSV}'
    assert os.path.isdir(config.IMAGES_DIR), f'Images directory not found: {config.IMAGES_DIR}'

    labels_df = pd.read_csv(config.LABELS_CSV)
    label_mapping = {'NON-SELF-HARM': 0, 'SELF-HARM': 1}

    if 'Label Akhir' not in labels_df.columns or 'filename' not in labels_df.columns:
        raise ValueError("CSV must have columns 'Label Akhir', 'filename, and 'Teks Terlihat (optional)'")
    
    labels_df = labels_df[labels_df['Label Akhir'].isin(label_mapping.keys())].copy()
    labels_df['Label'] = labels_df['Label Akhir'].map(label_mapping)

    print(f"Total samples after filtering: {len(labels_df)}")
    print(f'Overall class distribution:')
    for label, count in labels_df['Label'].value_counts().sort_index().items():
        label_name = 'NON-SELF-HARM' if label == 0 else 'SELF-HARM'
        print(f' {label} ({label_name}); {count} ({count/len(labels_df)*100:.1f}%)')

    train_df, val_df = train_test_split(
        labels_df,
        test_size =  0.2,
        random_state = config.SEED,
        stratify = labels_df['Label']
    )

    # Inverse-frequency class weights from train split: total/(num_classes*count_i)
    class_counts = train_df['Label'].value_counts().to_dict()
    class_weights = []
    total_train = len(train_df)
    for class_idx in range(config.NUM_CLASSES):
        count = class_counts.get(class_idx, 1)
        class_weights.append(total_train / (config.NUM_CLASSES * count))
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    print(f'\nTrain set: {len(train_df)} samples')
    for label, count in train_df['Label'].value_counts().sort_index().items():
        label_name = 'NON-SELF-HARM' if label == 0 else 'SELF-HARM'
        print(f' {label} ({label_name}); {count} ({count/len(train_df)*100:.1f}%)')

    print(f'\nValidation set: {len(val_df)} samples')
    for label, count in val_df['Label'].value_counts().sort_index().items():
        label_name = 'NON-SELF-HARM' if label == 0 else 'SELF-HARM'
        print(f' {label} ({label_name}); {count} ({count/len(val_df)*100:.1f}%)')

    clip_processor = CLIPImageProcessor.from_pretrained(config.CLIP_MODEL_NAME)
    try:
        electra_tokenizer = AutoTokenizer.from_pretrained(config.ELECTRA_MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load tokenizer for ELECTRA_MODEL_NAME='{config.ELECTRA_MODEL_NAME}'. "
            "Check repo id spelling, model visibility (public/private), and Hugging Face auth (hf auth login)."
        ) from e

    train_ds = MultimodalDataset(
        train_df, config.IMAGES_DIR, electra_tokenizer, clip_processor, max_len = config.MAX_LEN, is_train = True
    )
    val_ds = MultimodalDataset(
        val_df, config.IMAGES_DIR, electra_tokenizer, clip_processor, max_len = config.MAX_LEN, is_train = False
    )
    train_loader =  DataLoader(
        train_ds, batch_size = config.BATCH_SIZE, shuffle = True, num_workers = config.NUM_WORKERS,
        collate_fn = MultimodalDataset.collate_batch, pin_memory = torch.cuda.is_available()
    )
    val_loader =  DataLoader(
        val_ds, batch_size = config.BATCH_SIZE, shuffle = False, num_workers = config.NUM_WORKERS,
        collate_fn = MultimodalDataset.collate_batch, pin_memory = torch.cuda.is_available()
    )

    return train_loader, val_loader, clip_processor, electra_tokenizer, class_weights


def  main():
    parser = argparse.ArgumentParser(description =  'Train Multimodal Fusion Model')
    parser.add_argument('--images_dir', type=str, default=None, 
                        help='Path to images directory')
    parser.add_argument('--labels_csv', type=str, default=None, 
                        help='Path to labels CSV file')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None, 
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=None, 
                        help='Learning rate for trainable layers')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma for Focal Loss (higher = focus more on hard samples)')
    parser.add_argument('--clip_model_name', type=str, default=None,
                        help='Hugging Face model id for CLIP (default from Config)')
    parser.add_argument('--electra_model_name', type=str, default=None,
                        help='Hugging Face model id for ELECTRA (default from Config)')
    parser.add_argument('--wandb_project', type=str, default='thesis-elsa', 
                        help='W&B project name')
    parser.add_argument('--wandb_name', type=str, default=None, 
                        help='W&B run name')
    parser.add_argument('--wandb_verbose', action='store_true',
                        help='Show verbose W&B console logs (default is quiet mode)')
    parser.add_argument('--upload_wandb_artifact', action='store_true',
                        help='Upload best model checkpoint as W&B artifact (can be slow and verbose)')
    parser.add_argument('--no_wandb', action='store_true', 
                        help='Disable W&B logging')
    parser.add_argument('--notes', type=str, default=None, 
                        help='Notes for this training run')
    parser.add_argument('--tags', type=str, nargs='+', default=None, 
                        help='Tags for W&B run (e.g. --tags thesis experiment1)')
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
    if args.learning_rate is not None:
        config.LEARNING_RATE = args.learning_rate
    if args.clip_model_name:
        config.CLIP_MODEL_NAME = args.clip_model_name
    if args.electra_model_name:
        config.ELECTRA_MODEL_NAME = args.electra_model_name
    
    # Mengatur seed untuk reproducibility
    set_seed(config.SEED)
    os.makedirs(config.RESULT_DIR, exist_ok = True)
    
    # Save training notes to local file 
    if args.notes:
        notes_file = os.path.join(config.RESULT_DIR, 'training_notes.txt')
        with open(notes_file, 'a', encoding='utf-8') as f:
            from datetime import datetime
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Run Name: {args.wandb_name or 'default'}\n")
            f.write(f"Notes: {args.notes}\n")
            if args.tags:
                f.write(f"Tags: {', '.join(args.tags)}\n")
        print(f'Notes disimpan ke: {notes_file}') 

    # Initialize W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        if not args.wandb_verbose:
            os.environ['WANDB_SILENT'] = 'true'
            os.environ['WANDB_CONSOLE'] = 'off'

        wandb_run_name = args.wandb_name if args.wandb_name else 'nama_default' 

        wandb.init(
            project =  args.wandb_project,
            name = wandb_run_name,
            notes = args.notes,
            tags = args.tags,
            settings = wandb.Settings(console='off' if not args.wandb_verbose else 'auto'),
            config = {
                'model_name': config.MODEL_NAME,
                'batch_size': config.BATCH_SIZE,
                'epochs': config.EPOCHS,
                'learning_rate': config.LEARNING_RATE,
                'max_len': config.MAX_LEN,
                'num_classes': config.NUM_CLASSES,
                'image_size': config.IMAGE_SIZE,
                'seed': config.SEED,
                'patience': config.PATIENCE,
                'clip_model': config.CLIP_MODEL_NAME,
                'electra_model': config.ELECTRA_MODEL_NAME,
                'fusion_text_dim': config.FUSION_TEXT_DIM
            }
        )

    print(f'Model: {config.MODEL_NAME}')
    print(f'Device: {config.DEVICE}')
    print(f'Result directory: {config.RESULT_DIR}')
    print(f'CSV Path: {config.LABELS_CSV}')
    print(f'Images directory: {config.IMAGES_DIR}')

    train_loader, val_loader, clip_processor, electra_tokenizer, class_weights = load_data(config)

    clip_model = CLIPVisionModelWithProjection.from_pretrained(config.CLIP_MODEL_NAME)
    try:
        electra_model = AutoModel.from_pretrained(config.ELECTRA_MODEL_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model for ELECTRA_MODEL_NAME='{config.ELECTRA_MODEL_NAME}'. "
            "If this is a private/gated model, login first with: hf auth login"
        ) from e

    print('Initializing fusion model...')
    model = CLIPElectraFusion(
        clip_model = clip_model,
        electra_model = electra_model,
        fusion_text_dim = config.FUSION_TEXT_DIM,
        num_classes = config.NUM_CLASSES,
        freeze_encoders = True
    )

    model = model.to(config.DEVICE)

    analyze_model_parameters(model)

    optimizer = setup_optimizer(model, config.LEARNING_RATE)
    scheduler = setup_scheduler(optimizer)
    print(f'Class weights (train split): {class_weights.tolist()}')
    print(f'Using Focal Loss with gamma={args.focal_gamma}')
    criterion = FocalLoss(alpha=class_weights.to(config.DEVICE), gamma=args.focal_gamma)
    early_stopper = EarlyStopping(patience = config.PATIENCE, mode = 'max')

    history = {
        'train_loss' : [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lrs': []
    }

    #Output filenames with model name
    best_model_path = os.path.join(config.RESULT_DIR, f'bestmodel_{config.MODEL_NAME}.pth')
    best_f1 = -1.0
    best_epoch = 0

    print('\nStarting training...')

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

        print(f'Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f}')
        print(f'Val Loss : {val_loss:.4f} | Val Acc : {val_acc:.4f}')
        print(f'Precision : {p:.4f} | Recall : {r:.4f} | F1 Score : {f1:.4f}')
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
                'val/f1_score': f1,
                'learning_rate': current_lr
            })
        improved = early_stopper.step(f1)

        if improved:
            print(f'Best model saved (F1 Score improved to {f1:.4f})')
            # Save checkpoint with model name in filename
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1_score': f1,
                'history': history
            }, best_model_path)
            best_f1 = f1
            best_epoch = epoch

            # log best model to W&B
            if use_wandb:
                wandb.run.summary['best_f1'] = f1
                wandb.run.summary['best_epoch'] = epoch
        else:
            print(f'No improvement ({early_stopper.num_bad}/{config.PATIENCE})')

        if early_stopper.should_stop:
            print(f'Early stopping triggered after {epoch} epochs')
            break

        print()

    print(f'\nTraining completed. Best F1 Score: {best_f1:.4f} at epoch {best_epoch}')
    print(f'Best model saved to: {best_model_path}')

    print('Loading best model for final evaluation...')
    checkpoint = torch.load(best_model_path, map_location = config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    _, val_acc, p, r, f1, cm, val_preds, val_trues = evaluate(
        model, val_loader, criterion, config.DEVICE
    )

    print('\nFinal Evaluation Results:')
    print(f'Acc : {val_acc:.4f}')
    print(f'Precision : {p:.4f}')
    print(f'Recall : {r:.4f}')
    print(f'F1 Score : {f1:.4f}')

    print_classification_report(val_trues, val_preds)

    #save plots with model name
    cm_path = os.path.join(config.RESULT_DIR, f'confusion_matrix_{config.MODEL_NAME}.png')
    plot_confusion_matrix(cm, save_path = cm_path)

    history_path = os.path.join(config.RESULT_DIR, f'training_history_{config.MODEL_NAME}.png')
    plot_training_history(history, save_path = history_path)

    print(f'\nPlots saved to {config.RESULT_DIR}')

    #log final artifacts to W&B
    if use_wandb:
        wandb.log({
            'confusion_matrix': wandb.Image(cm_path),
            'training_history': wandb.Image(history_path)
        })

        #save model artifact to W&B only when explicitly requested
        if args.upload_wandb_artifact:
            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(best_model_path)
            wandb.log_artifact(artifact)
            print('W&B model artifact uploaded.')
        else:
            print('Skipping W&B model artifact upload (use --upload_wandb_artifact to enable).')

        wandb.finish()
        print('W&B artifacts logged and run finished.')

if __name__ == '__main__':
    main()
                      