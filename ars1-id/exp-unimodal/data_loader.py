import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MultiModalDataset(Dataset):
    def __init__(self, df, images_dir, electra_tokenizer, clip_processor,
                 max_len=128, is_train=False, mode='text_only'):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.electra_tokenizer = electra_tokenizer
        self.clip_processor = clip_processor
        self.max_len = max_len
        self.mode = mode  # 'text_only' | 'image_only'

        if is_train and mode == 'image_only':
            self.augmentation = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.augmentation = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        label = int(row['Label']) if 'Label' in row else int(row['Label Akhir'])

        # --- Gambar ---
        if self.mode == 'image_only':
            fname = row['filename']
            img_path = os.path.join(self.images_dir, fname)
            image = Image.open(img_path).convert('RGB')
            if self.augmentation:
                image = self.augmentation(image)
            clip_inputs = self.clip_processor(images=image, return_tensors='pt')
            pixel_values = clip_inputs['pixel_values'].squeeze(0)
        else:
            # text_only: dummy, tidak dikirim ke encoder
            pixel_values = torch.zeros(3, 224, 224)

        # --- Teks ---
        if self.mode == 'text_only':
            text = str(row['Teks Terlihat']) if 'Teks Terlihat' in row else ''
            enc = self.electra_tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            input_ids = enc['input_ids'].squeeze(0)
            attention_mask = enc['attention_mask'].squeeze(0)
        else:
            # image_only: dummy
            input_ids = torch.zeros(self.max_len, dtype=torch.long)
            attention_mask = torch.zeros(self.max_len, dtype=torch.long)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': torch.tensor(label, dtype=torch.long)
        }


def collate_batch(batch):
    pixel_values = torch.stack([b['pixel_values'] for b in batch], dim=0)
    input_ids = torch.stack([b['input_ids'] for b in batch], dim=0)
    attention_mask = torch.stack([b['attention_mask'] for b in batch], dim=0)
    labels = torch.stack([b['label'] for b in batch], dim=0)
    
    return {
        'pixel_values': pixel_values,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'label': labels
    }
