"""
Training module for the multimodal fusion model.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    preds, trues = [], []
    
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        
        pixel = batch['pixel_values'].to(device)
        ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        logits, _, _ = model(pixel, ids, mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds.extend(logits.argmax(-1).detach().cpu().tolist())
        trues.extend(labels.detach().cpu().tolist())
    
    acc = accuracy_score(trues, preds)
    return total_loss / len(loader), acc


def setup_optimizer(model, lr_pretrain=1e-5, lr_head=5e-4, weight_decay=1e-2):
    pretrained_params = [
        p for n, p in model.named_parameters() 
        if ('clip' in n or 'electra' in n) and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters() 
        if ('clip' not in n and 'electra' not in n) and p.requires_grad
    ]
    
    optimizer = torch.optim.AdamW([
        {'params': pretrained_params, 'lr': lr_pretrain},
        {'params': head_params, 'lr': lr_head}
    ], weight_decay=weight_decay)
    
    return optimizer


def setup_scheduler(optimizer, mode='min', factor=0.5, patience=1):

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=mode, 
        factor=factor, 
        patience=patience
    )
    return scheduler
