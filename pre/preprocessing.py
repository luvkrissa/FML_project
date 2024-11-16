import pandas as pd
import os
import shutil
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from pathlib import Path
import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import torch.utils.data as data

OLIVES_dir = "/storage/ice1/shared/d-pace_community/makerspace-datasets/MEDICAL/OLIVES/OLIVES"
Labels_dir = "/home/hice1/mzhou346/Project/OLIVES_Dataset_Labels/ml_centric_labels"
Work_dir = "/home/hice1/mzhou346/Project"

labels_file = os.path.join(Labels_dir, "Biomarker_Clinical_Data_Images.csv") 
df = pd.read_csv(labels_file)

eye_ids = df["Eye_ID"].unique()

# Split patient IDs into train and test
train_ids, test_ids = train_test_split(eye_ids, test_size=0.2, random_state=42)

# Split the labels dataframe into train and test based on patient IDs
train_labels = df[df["Eye_ID"].isin(train_ids)]
test_labels = df[df["Eye_ID"].isin(test_ids)]

# Save the splits to new CSV files for easier access
train_labels.to_csv(os.path.join(Work_dir, "train_labels.csv"), index=False)
test_labels.to_csv(os.path.join(Work_dir, "test_labels.csv"), index=False)

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
        b1 = self.df.iloc[idx,1]
        b2 = self.df.iloc[idx,2]
        b3 = self.df.iloc[idx,3]
        b4 = self.df.iloc[idx, 4]
        b5 = self.df.iloc[idx, 5]
        b6 = self.df.iloc[idx, 6]
        bio_tensor = torch.tensor([b1, b2, b3, b4, b5, b6])
        return image, bio_tensor

# construct data loader
mean = (.1706)
std = (.2112)

normalize = transforms.Normalize(mean=mean, std=std)

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),

    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    normalize,
])

train_dataset = OLIVES(os.path.join(Work_dir, "train_labels.csv"),OLIVES_dir,transforms = train_transform)
test_dataset = OLIVES(os.path.join(Work_dir, "test_labels.csv"),OLIVES_dir,transforms = val_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=128, shuffle=True,
    num_workers=4, pin_memory=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False,
    num_workers=0, pin_memory=True,drop_last=False)

print("done")