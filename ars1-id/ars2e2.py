import torch
import torch.nn as nn


class CLIPElectraMLPFusion(nn.Module):
    def __init__(self, clip_model, electra_model,
                 fusion_text_dim=256,
                 num_classes=2, freeze_encoders=True):
        super().__init__()
        self.clip = clip_model
        self.electra = electra_model

        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False
            for p in self.electra.parameters():
                p.requires_grad = False

        # CLIP dimensi output image feature
        self.img_dim = clip_model.config.projection_dim  # 512

        # ELECTRA dimensi output text feature
        electra_hidden_dim = electra_model.config.hidden_size  # e.g. 768

        # Projection layer text: (electra_hidden_dim -> fusion_text_dim : 256)
        self.project_text = nn.Sequential(
            nn.Linear(electra_hidden_dim, fusion_text_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_text_dim)
        )

        # Fusion dimension: 512 (CLIP) + 256 (text projected) = 768
        self.fusion_dim = self.img_dim + fusion_text_dim

        # 3-layer MLP classifier (langsung dari concat fitur)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, self.fusion_dim // 2),  # 768 -> 384
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(self.fusion_dim // 2, self.fusion_dim // 4),  # 384 -> 192
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.fusion_dim // 4, num_classes)  # 192 -> num_classes
        )

    def forward(self, pixel_values, input_ids, attention_mask):

        # --- Image Encoder (CLIP) ---
        img_output = self.clip.get_image_features(pixel_values)
        if hasattr(img_output, 'pooler_output'):
            img_feats = img_output.pooler_output
        elif isinstance(img_output, torch.Tensor):
            img_feats = img_output
        else:
            img_feats = img_output[0] if isinstance(img_output, (tuple, list)) else img_output

        # L2 normalization
        img_proj = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)
        # Shape: (batch_size, 512)

        # --- Text Encoder (ELECTRA) ---
        txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = txt_out.last_hidden_state
        # Shape: (batch_size, seq_len, 768)

        # Mean pooling (exclude padding tokens)
        attn = attention_mask.unsqueeze(-1).float()
        sum_emb = (last_hidden * attn).sum(dim=1)
        sum_mask = attn.sum(dim=1).clamp(min=1e-9)
        text_emb = sum_emb / sum_mask
        # Shape: (batch_size, 768)

        # Project text: 768 -> 256
        text_proj = self.project_text(text_emb)
        # Shape: (batch_size, 256)

        # --- Fusion: Concatenate image + text ---
        fused = torch.cat([img_proj, text_proj], dim=-1)
        # Shape: (batch_size, 768)

        # --- MLP Classifier ---
        logits = self.classifier(fused)

        return logits, img_proj, text_proj


class EarlyStopping:

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
