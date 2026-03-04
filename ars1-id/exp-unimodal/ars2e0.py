"ARSITEKTUR B SIMPEL FUSION, TEXT ONLY + MLP CLASSIFIER" 

import torch
import torch.nn as nn


class TextOnlyElectra(nn.Module):

    def __init__(self, clip_model, electra_model,
                 fusion_text_dim=256,
                 num_classes=2, freeze_encoders=True,
                 num_unfreeze_last_electra_layers=4):
        super().__init__()

        # CLIP disimpan tapi dibekukan total dan TIDAK digunakan di forward
        # clip_model boleh None untuk mode text_only (tidak perlu di-load)
        self.clip = clip_model
        if self.clip is not None:
            for p in self.clip.parameters():
                p.requires_grad = False

        # ELECTRA  encoder ang aktif
        self.electra = electra_model
        if freeze_encoders:
            for p in self.electra.parameters():
                p.requires_grad = False
        
        # Unfreeze sebagian layer terakhir ELECTRA saja jika diinginkan
        # num_unfreeze_last_electra_layers=2 akan unfreeze layer 10-11 dari 12 layer (0-indexed)
        if num_unfreeze_last_electra_layers > 0:
            num_total_layers = self.electra.config.num_hidden_layers  # 12
            layers_to_unfreeze = list(range(num_total_layers - num_unfreeze_last_electra_layers, num_total_layers))
            for layer_idx in layers_to_unfreeze:
                for p in self.electra.encoder.layer[layer_idx].parameters():
                    p.requires_grad = True
            print(f"✓ Unfroze ELECTRA layers: {layers_to_unfreeze}")

        electra_hidden_dim = electra_model.config.hidden_size  # 768

        # Projection: 768 -> fusion_text_dim (256)
        self.project_text = nn.Sequential(
            nn.Linear(electra_hidden_dim, fusion_text_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_text_dim)
        )

        # 3-layer MLP classifier (sama seperti di CLIPElectraFusion tapi input = fusion_text_dim)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_text_dim, fusion_text_dim // 2),        # 256 -> 128
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_text_dim // 2, fusion_text_dim // 4),   # 128 -> 64
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(fusion_text_dim // 4, num_classes)              # 64 -> num_classes
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # pixel_values diterima tapi TIDAK diproses (saluran gambar dinonaktifkan)

        # Extract text features dari ELECTRA
        txt_out = self.electra(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = txt_out.last_hidden_state
        # Shape: (batch_size, sequence_length, 768)

        # Mean pooling (sama seperti di CLIPElectraFusion)
        attn = attention_mask.unsqueeze(-1).float()
        sum_emb = (last_hidden * attn).sum(dim=1)
        sum_mask = attn.sum(dim=1).clamp(min=1e-9)
        text_emb = sum_emb / sum_mask
        # Shape: (batch_size, 768)

        # Project text 768 -> 256
        text_proj = self.project_text(text_emb)
        # Shape: (batch_size, 256)

        # Langsung ke MLP classifier (tanpa fusion transformer)
        logits = self.classifier(text_proj)

        # Return logits + None placeholder untuk img_proj agar kompatibel dengan train loop
        return logits, None, text_proj


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
