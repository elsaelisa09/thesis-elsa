"ARSITEKTUR B SIMPEL FUSION, IMGE ONLY (CLIP) + MLP CLASSIFIER"

import torch
import torch.nn as nn


class ImageOnlyCLIP(nn.Module):

    def __init__(self, clip_model, electra_model,
                 fusion_img_dim=256,
                 num_classes=2, freeze_encoders=True):
        super().__init__()

        # CLIP  encoder  aktif
        self.clip = clip_model
        if freeze_encoders:
            for p in self.clip.parameters():
                p.requires_grad = False

        # ELECTRA disimpan & dibekukan total dan tidak  digunakan di forward
        self.electra = electra_model
        for p in self.electra.parameters():
            p.requires_grad = False

        self.img_dim = clip_model.config.projection_dim  # 512

        # Projection: 512 -> fusion_img_dim (256)
        self.project_img = nn.Sequential(
            nn.Linear(self.img_dim, fusion_img_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_img_dim)
        )

        # 3-layer MLP classifier (input = fusion_img_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_img_dim, fusion_img_dim // 2),        # 256 -> 128
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_img_dim // 2, fusion_img_dim // 4),   # 128 -> 64
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_img_dim // 4, num_classes)              # 64 -> num_classes
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # input_ids dan attention_mask diterima tapi TIDAK diproses (saluran teks dinonaktifkan)

        # Extract image features dari CLIP
        img_output = self.clip.get_image_features(pixel_values)
        if hasattr(img_output, 'pooler_output'):
            img_feats = img_output.pooler_output
        elif isinstance(img_output, torch.Tensor):
            img_feats = img_output
        else:
            img_feats = img_output[0] if isinstance(img_output, (tuple, list)) else img_output

        # L2 normalization (sama seperti di CLIPElectraFusion)
        img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-10)
        # Shape: (batch_size, 512)

        # Project image 512 -> 256
        img_proj = self.project_img(img_norm)
        # Shape: (batch_size, 256)

        # Langsung ke MLP classifier (tanpa fusion transformer)
        logits = self.classifier(img_proj)

        # Return logits + img_proj + None placeholder untuk text_proj agar kompatibel dengan train loop
        return logits, img_proj, None


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
