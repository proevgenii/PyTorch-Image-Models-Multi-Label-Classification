import pickle5 as pickle
import os
import argparse
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import CustomData, pil_loader, seed_everything
from torchvision import transforms
import torch 
import pandas as pd
from PIL import Image
seed_everything(42)

class ImageDataset(Dataset):
    """ Leaf Disease Dataset """
    def __init__(self,
                image_names,
                labels,
                image_dir, 
                transforms):        
        self.image_names = image_names
        self.image_dir = image_dir
        self.transforms = transforms                
        self.labels = labels

    def __len__(self) -> int:
        return len(self.image_names)
    
    def __getitem__(self, idx: int):
        image = Image.open(os.path.join(self.image_dir, self.image_names[idx]))   
        return self.transforms(image), self.labels[idx]

def get_dataloaders_kaggle(name='base', size = 224,
                    mean = (0.485, 0.456, 0.406),
                    std = (0.229, 0.224, 0.225),
                   add_synt = False,
                   n_workers=16):
    IMG_SIZE = size
    BATCH = 64
    N_WORKERS = n_workers
    MEAN = mean
    STD = std
    PATH = '/raid/eprosvirin/img_clf/kaggle_data/'
    train_transform = transforms.Compose( # 'category'
                                [
                                transforms.RandomHorizontalFlip(p=0.5,),
                                transforms.RandomApply(torch.nn.ModuleList([
                                    transforms.RandomRotation(degrees=(-7, 7)),
                                    transforms.RandomAdjustSharpness(sharpness_factor=2)]), p=0.3),    
                                transforms.ToTensor(),
                                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                transforms.Normalize(MEAN, STD)
                                ])
    test_transform = transforms.Compose(
                                [
                                transforms.ToTensor(),
                                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                                transforms.Normalize(MEAN, STD)
                                ])
    train_df = pd.read_csv(PATH+'train_df.csv').sample(300)
    test_df = pd.read_csv(PATH+'test_df.csv').sample(300)
    print('TRAIN SHAPE:', train_df.shape, 'TEST SHAPE:', test_df.shape)
    LABELS = ['complex', 'healthy', 'powdery_mildew', 'rust',
       'frog_eye_leaf_spot', 'scab']
    print("KAGGLE DATA: train shaep: {train_df.shape} test_df shape: {test_df.shape}")
    
    trainset = ImageDataset(image_names=train_df.image.values, 
                                labels=train_df[LABELS].values, 
                                image_dir=PATH+'train_images/', 
                                transforms=train_transform)
    
    testset = ImageDataset(image_names=test_df.image.values, 
                                labels=test_df[LABELS].values, 
                                image_dir=PATH+'train_images/', 
                                transforms=test_transform)

    dataloaders = {'train': DataLoader(trainset, batch_size=BATCH, shuffle=True, num_workers=N_WORKERS),
                   'test': DataLoader(testset, batch_size=BATCH, num_workers=N_WORKERS)}
    return dataloaders