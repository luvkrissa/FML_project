import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch


class OLIVES(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        img_dir = img_dir.strip()
        df = df.strip()
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
        self.df = self.df.dropna(subset=self.df.columns[2:21])  # Keep only valid rows
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx,0]
        image = Image.open(path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        bio_values = [self.df.iloc[idx, col] for col in range(2, 12)]  # Extract 
        clinical_values = [self.df.iloc[idx, col] for col in range(18, 22)]

        # Convert bio values to tensor
        bio_tensor = torch.tensor(bio_values, dtype=torch.float32)
        clinical_tensor = torch.tensor(clinical_values, dtype=torch.float32)
        return image, bio_tensor, clinical_tensor




class RECOVERY(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx,0]
        image = Image.open(path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        return image



class RECOVERY_TEST(data.Dataset):
    def __init__(self,df, img_dir, transforms):
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx,0]
        image = Image.open(path).convert("L")
        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        b1 = self.df.iloc[idx,1]
        b2 = self.df.iloc[idx,2]
        b3 = self.df.iloc[idx,3]
        b4 = self.df.iloc[idx, 4]
        b5 = self.df.iloc[idx, 5]
        b6 = self.df.iloc[idx, 6]
        bio_tensor = torch.tensor([b1, b2, b3, b4, b5, b6])
        return image, bio_tensor

