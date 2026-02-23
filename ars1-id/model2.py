"""
Model architecture 2 - Alternative fusion approach.
TODO: Implement alternative architecture here
"""

import torch
import torch.nn as nn


class CLIPElectraFusion(nn.Module):
    """
    Alternative fusion architecture - Model 2
    TODO: Implement different fusion strategy
    """
    
    def __init__(self, clip_model, electra_model,
                 fusion_img_dim=512, fusion_text_dim=256,
                 num_classes=2, freeze_encoders=True):
        super().__init__()
        self.clip = clip_model
        self.electra = electra_model

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.electra.parameters():
                p.requires_grad = False

        # TODO: Implement your architecture here
        # This is a placeholder - implement your custom fusion logic
        
        self.img_dim = 512
        self.fusion_text_dim = fusion_text_dim
        
        # Example placeholder layers
        self.project_text = nn.Sequential(
            nn.Linear(768, fusion_text_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_text_dim)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.img_dim + fusion_text_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Forward pass - customize based on your architecture
        Must return: logits, img_features, text_features
        """
        # TODO: Implement forward pass
        
        # Placeholder implementation
        img_feats = self.clip.get_image_features(pixel_values)
        img_proj = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)
        
        txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = txt_out.last_hidden_state
        
        attn = attention_mask.unsqueeze(-1).float()
        sum_emb = (last_hidden * attn).sum(dim=1)
        sum_mask = attn.sum(dim=1).clamp(min=1e-9)
        text_emb = sum_emb / sum_mask
        
        text_proj = self.project_text(text_emb)
        
        # Simple concatenation
        fused = torch.cat([img_proj, text_proj], dim=-1)
        logits = self.classifier(fused)
        
        return logits, img_proj, text_proj


class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience=3, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best = None
        self.num_bad = 0
        self.should_stop = False
        
    def step(self, value):
        if self.best is None:
            self.best = value
            self.num_bad = 0
            return True
            
        improve = (value > self.best) if self.mode == 'max' else (value < self.best)
        
        if improve:
            self.best = value
            self.num_bad = 0
            return True
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
            return False
