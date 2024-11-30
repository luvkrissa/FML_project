import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch


class OLIVES(data.Dataset):
    def __init__(self,df, img_dir, transforms, opt):
        img_dir = img_dir.strip()
        df = df.strip()
        self.img_dir = img_dir
        self.transforms = transforms
        self.df = pd.read_csv(df)
        self.df = self.df.dropna(subset=self.df.columns[2:17])  # Keep only valid rows
        self.opt = opt
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.img_dir + self.df.iloc[idx,0]
        image = Image.open(path).convert("L")
        if self.opt.model in ['maxvit_base_tf_224', 'vit_b_16', 'maxvit_tiny_tf_224']:
            image = image.convert("RGB")  # Convert to RGB directly using PIL

        image = np.array(image)
        image = Image.fromarray(image)
        image = self.transforms(image)
        if self.opt.model == 'resnet50':
            bio_values = [self.df.iloc[idx, col] for col in range(2, 8)]  # Extract 
        else:
            bio_values = [self.df.iloc[idx, col] for col in range(2, 8)]

        # Convert bio values to tensor
        bio_tensor = torch.tensor(bio_values, dtype=torch.float32)
        return image, bio_tensor




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

