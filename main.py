import pandas as pd 
import numpy as np 
import os 
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import train_test_split

import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import pretrainedmodels # download pretraindemodels from https://www.kaggle.com/abhishek/pretrained-models/downloads/pretrained-models.zip/1

import kappa_optimizer

import time 
from PIL import Image
train_on_gpu = True

import cv2
import albumentations
from albumentations import torch as AT

import gc

device = torch.device('cuda')

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def prepare_labels(y):
    # From here: https://www.kaggle.com/pestipeti/keras-cnn-starter
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder

class GlassDataset(Dataset):
    def __init__(self, df, datatype='train', transform = transforms.Compose([transforms.CenterCrop(32), transforms.ToTensor()]), y=None):
        self.df = df
        self.datatype=datatype
        if self.datatype=='train':
            self.image_file_list = [f'C:/Data/slepota_data/train/{i}.png' for i in df['id_code'].values]
            self.labels=y
        else:
            self.image_file_list = [f'C:/Data/slepota_data/test/{i}.png' for i in df['id_code'].values]
            self.labels = np.zeros((df.shape[0], 5))
        self.transform = transform

    def __len__(self):
        return len(self.image_file_list)

    def __getitem__(self, idx):
        img_name=self.image_file_list[idx]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = self.transform(image = img)

        img_name_short = self.image_file_list[idx].split('.')[0]

        label = self.labels[idx]
        if self.datatype==' test':
            return image, label, img_name
        else:
            return image, label




def main():
    train = pd.read_csv('C:/Data/slepota_data/train.csv')
    test = pd.read_csv('C:/Data/slepota_data/test.csv')
    sample_submission = pd.read_csv('C:/Data/slepota_data/sample_submission.csv')
    y, le = prepare_labels(train['diagnosis'])

    data_transforms = albumentations.Compose([
        albumentations.Resize(224,224),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightness(),
        albumentations.ShiftScaleRotate(rotate_limit=15, scale_limit=0.10),
        albumentations.JpegCompression(80),
        albumentations.HueSaturationValue(),
        albumentations.Normalize(),
        AT.ToTensor()
    ])

    test_transforms = albumentations.Compose([
        albumentations.Resize(224,224),
        albumentations.HorizontalFlip(),
        albumentations.Normalize(),
        AT.ToTensor(),
    ])#Возможно добавить больше test time augmentation

    train_set = GlassDataset(df=train, datatype='train', transform=data_transforms, y=y)
    test_set = GlassDataset(df=test, datatype='test', transform=test_transforms)

    tr, val = train_test_split(train.diagnosis, stratify = train.diagnosis, test_size=0.1)

    train_sampler = SubsetRandomSampler(list(tr.index))
    validation_sampler = SubsetRandomSampler(list(val.index))

    batch_size = 64
    num_workers = 0
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = DataLoader(train_set, batch_size=batch_size, sampler=validation_sampler, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    model = pretrainedmodels.__dict__['resnet101'](pretrained=None)

    model.avg_pool = nn.AdaptiveAvgPool2d(1)
    model.last_linear = nn.Sequential(
                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.25),
                          nn.Linear(in_features=2048, out_features=2048, bias=True),
                          nn.ReLU(),
                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.5),
                          nn.Linear(in_features=2048, out_features=1, bias=True),
                         )
    model.load_state_dict(torch.load("../input/mmmodel/model.bin"))
    model = model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.eval()

    valid_predictions = np.zeros((len(val), 1))
    tk_valid = tqdm(valid_loader)
    for i, x_batch in enumerate(tk_valid):
        x_batch = x_batch["image"]
        pred = model(x_batch.to(device))
        valid_predictions[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_preds1 = np.zeros((len(test_set), 1))
    tk0 = tqdm(test_loader)
    for i, x_batch in enumerate(tk0):
        x_batch = x_batch["image"]
        pred = model(x_batch.to(device))
        test_preds1[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

    test_preds2 = np.zeros((len(test_set), 1))
    tk0 = tqdm(test_loader)
    for i, x_batch in enumerate(tk0):
        x_batch = x_batch["image"]
        pred = model(x_batch.to(device))
        test_preds2[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

    test_preds3 = np.zeros((len(test_set), 1))
    tk0 = tqdm(test_loader)
    for i, x_batch in enumerate(tk0):
        x_batch = x_batch["image"]
        pred = model(x_batch.to(device))
        test_preds3[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

    test_preds4 = np.zeros((len(test_set), 1))
    tk0 = tqdm(test_loader)
    for i, x_batch in enumerate(tk0):
        x_batch = x_batch["image"]
        pred = model(x_batch.to(device))
        test_preds4[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

    test_preds5 = np.zeros((len(test_set), 1))
    tk0 = tqdm(test_loader)
    for i, x_batch in enumerate(tk0):
        x_batch = x_batch["image"]
        pred = model(x_batch.to(device))
        test_preds5[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)

    test_predictions = (test_preds1 + test_preds2 + test_preds3 + test_preds4 + test_preds5)/5.0

    optR = kappa_optimizer.OptimizerRounder()
    optR.fit(valid_predictions, y)
    coefficients = optR.coefficient()
    valid_predictions = optR.predict(valid_predictions, coefficients)
    test_predictions = optR.predict(test_predictions, coefficients)


    sample_submission.diagnosis = test_predictions.astype(int)
    sample_submission.to_csv("submission.csv", index=False)
    print('Validation prediction: ', valid_predictions)
    print('Validation score: ', metrics.cohen_kappa_score(y , valid_predictions, weights='quadratic'))


