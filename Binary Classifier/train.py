import random
import os
import numpy as np
from skimage import io
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from tqdm import tqdm
import csv
from torchvision import models
import torch.nn as nn
import glob

import cv2

def get_dataset(masks_dir):
    rust = []
    non_rust = []
    masks_paths = glob.glob(masks_dir+'*')
    for mask_path in masks_paths:
        mask = cv2.imread(mask_path, 0) 
        condition = (mask > 200)
        count = np.sum(condition)
        if count > 200:
            rust.append(mask_path.replace("masks", "images"))
        else:
            non_rust.append(mask_path.replace("masks", "images"))
    return rust, non_rust[:1683]

weights_directory = 'vgg16_300_patch_4thtry'
graph_file = 'vgg16_300_patch_graph_4thtry.csv'

os.makedirs(weights_directory, exist_ok=True)

model = models.vgg16(pretrained=True)
model.classifier[6] = nn.Linear(4096, 2)  # Change the last layer for binary classification

print("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

rust, non_rust = get_dataset('/home2/haseeb_muhammad/BinaryClassifier/Combined_images/train/masks/')
print("Initial Rust count:", len(rust))
print("Initial Non_Rust count:", len(non_rust))

random.seed(4)

rust_train = int(len(rust)*0.8)
rust_val = len(rust) - rust_train
non_rust_train = int(len(non_rust)*0.8)
non_rust_val = len(non_rust) - non_rust_train

rust_y_train = [1]*rust_train
rust_y_val = [1]*rust_val
non_rust_y_train = [0]*non_rust_train
non_rust_y_val = [0]*non_rust_val

print(f"rust_train:{rust_train} rust_y_train:{len(rust_y_train)} rust_val:{rust_val} rust_y_val:{len(rust_y_val)} non_rust_train:{non_rust_train} non_rust_y_train:{len(non_rust_y_train)} non_rust_val:{non_rust_val} non_rust_y_val:{len(non_rust_y_val)}")

x_train = rust[:rust_train]
x_train.extend(non_rust[:non_rust_train])
x_val = rust[:rust_val]
x_val.extend(non_rust[:non_rust_val])

y_train = rust_y_train
y_train.extend(non_rust_y_train)
y_val = rust_y_val
y_val.extend(non_rust_y_val)

x_train, x_val = np.array(x_train), np.array(x_val)
y_train, y_val = np.array(y_train), np.array(y_val)

print(f"x_train:{len(x_train)} x_val:{len(x_val)} y_train:{len(y_train)} y_val:{len(y_val)}")

transform_train = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((224, 224)),
                                     transforms.RandomHorizontalFlip(p=0.20),
                                     transforms.ColorJitter(brightness=0.25),
                                     transforms.GaussianBlur(5),
                                     transforms.ToTensor()
                                     ])
transform_val = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((224, 224)),
                                    transforms.ToTensor()])

class CData(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        img_path = self.x[index]
        image = io.imread(img_path)
        label = int(self.y[index])
        if self.transform:
            image = self.transform(image)
        return (image, label)


train_dataset = CData(x_train, y_train, transform=transform_train)
val_dataset = CData(x_val, y_val, transform=transform_val)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=4, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

weight_loc = f"/home2/haseeb_muhammad/BinaryClassifier/{weights_directory}"
os.makedirs(weight_loc, exist_ok=True)
step_train=0
step_val=0


for epoch in range(50):
  losses=[]
  acces=[]
  model.train()
  loop = tqdm(enumerate(train_loader), total=len(train_loader))
  for batch_idx, (data, targets) in loop:
    data = data.to(device)
    targets = targets.to(device)

    #Forward Pass
    pred = model(data)
    loss = criterion(pred, targets)
    losses.append(loss.item())
    acc = sum(pred.argmax(axis=1)==targets)/float(targets.shape[0])
    acces.append(acc.item())
    #Backward Pass
    optimizer.zero_grad()
    loss.backward()
    step_train+=1

    #step
    optimizer.step()
    loop.set_description("Epoch "+str(epoch)+" Acc "+str(np.mean(acces))+" Train Loss "+str(np.mean(losses)))

  train_acc = np.mean(acces)
  train_loss = np.mean(losses)
  torch.save(model.state_dict(), weight_loc+'/model_{}.pth'.format(epoch))

  model.eval()
  losses = []
  acces = []

  loop = tqdm(enumerate(val_loader), total=len(val_loader))
  with torch.no_grad():
    for batch_idx, (data, targets) in loop:
      data = data.to(device)
      targets = targets.to(device)

      #Forward Pass
      pred = model(data)
      loss = criterion(pred, targets)
      losses.append(loss.item())
      acc = sum(pred.argmax(axis=1)==targets)/float(targets.shape[0])
      acces.append(acc.item())
      loop.set_description("Epoch "+str(epoch)+" Acc "+str(np.mean(acces))+" Val Loss "+str(np.mean(losses)))
      step_val+=1

  with open(f"{graph_file}", 'a') as f:
      writer = csv.writer(f)
      writer.writerow([epoch, train_acc, train_loss, f'{np.mean(acces)}', f'{np.mean(losses)}'])