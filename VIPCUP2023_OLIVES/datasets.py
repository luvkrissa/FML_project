import torch.utils.data as data
from PIL import Image
import numpy as np
import pandas as pd
import os
import torch


class OLIVES(data.Dataset):
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
        b1 = self.df.iloc[idx,2]
        b2 = self.df.iloc[idx,3]
        b3 = self.df.iloc[idx,4]
        b4 = self.df.iloc[idx, 5]
        b5 = self.df.iloc[idx, 6]
        b6 = self.df.iloc[idx, 7]
        b7 = self.df.iloc[idx, 8]
        b8 = self.df.iloc[idx, 9]
        b9 = self.df.iloc[idx, 10]
        b10 = self.df.iloc[idx, 11]
        b11 = self.df.iloc[idx, 12]
        b12 = self.df.iloc[idx, 13]
        b13 = self.df.iloc[idx, 14]
        b14 = self.df.iloc[idx, 15]
        b15 = self.df.iloc[idx, 16]
        b16 = self.df.iloc[idx, 17]
        bio_tensor = torch.tensor([b1, b2, b3, b4, b5, b6,blabla])
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

