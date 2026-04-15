import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class_map = {
    "Non_Demented": 0,        # CN
    "Very_Mild_Demented": 1,  # MCI
    "Mild_Demented": 2,       # AD
    "Moderate_Demented": 2    # AD (merged)
}

class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.image_paths = self.df["filepath"].values
        self.labels_binary = self.df["label"].values.astype(np.int64)  # 🔥 IMPORTANT FIX

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels_binary[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR loading] {img_path} → {e}")
            # return a blank image instead of None (prevents crash)
            image = Image.new("RGB", (160, 160))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)