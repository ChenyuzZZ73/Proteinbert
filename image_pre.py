import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
#df = pd.read_table("label.txt",sep = " ", header=0)
#d  = df[["Bald", "Blond_Hair", "Eyeglasses", "Heavy_Makeup", "Male", "Sideburns", "Smiling", "Wavy_Hair", "Wearing_Hat", "Young"]].values
#print(d[:100])


class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""
    def __init__(self, txt_path, img_dir, transform=None):
        df = pd.read_csv(txt_path, sep=" ", header=0)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df["img"].values
        self.y = df[["Bald", "Blond_Hair", "Eyeglasses", "Heavy_Makeup", "Male", "Sideburns", "Smiling", "Wavy_Hair", "Wearing_Hat", "Young"]].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))

        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]

class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_c, out_c, 5, 2, padding=2, output_padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.convt(x)
        out = self.norm(out)
        return self.relu(out)

class DownBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 5, 2, 2, bias=False)
        self.norm = nn.InstanceNorm2d(out_c, affine=True)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.lrelu(out)

class Discriminator(nn.Module):
    def __init__(self,nc=3,ndf=64):

        super().__init__()

        self.encode_img = nn.Sequential(
            nn.Conv2d(nc, ndf * 2, 5, 2, 2, bias=False),
            # nn.Linear(in_features=nz + 10, out_features=ngf * 16 * 4 * 4),
            nn.BatchNorm2d(ndf * 2),
            nn.ReLU(),
            nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ReLU(),
            nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.ReLU(),
        )  # (ngf * 8 * 4 *4)

        self.fc = nn.Sequential(
            # nn.ConvTranspose2d(nz + 10, ngf * 16, 5, 2, 1, bias=False),
            nn.Linear(in_features=10, out_features=ndf * 8 * 4 * 4),
            nn.BatchNorm1d(ndf * 8 * 4 * 4),
            nn.ReLU())

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 32, 1, 4))


        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.Conv2d(nc, ndf, 5, 2, 2, bias=False),
                #nn.Conv2d(nc + 10, ndf, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf),
                #nn.InstanceNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (B, nch, 128, 128) -> (B, nch_d, 64, 64)
            'layer1': nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
                #nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf * 2),
                nn.InstanceNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (B, nch_d, 64, 64) -> (B, nch_d*2, 32, 32)
            'layer2': nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
                #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf * 4),
                nn.InstanceNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (B, nch_d*2, 32, 32) -> (B, nch_d*4, 16, 16)
            'layer3': nn.Sequential(
                nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
                #nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf * 8),
                nn.InstanceNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (B, nch_d*4, 16, 16) -> (B, nch_g*8, 8, 8)
            'layer4':   nn.Sequential(
                nn.Conv2d(ndf*8, ndf * 16,5,2,2,bias=False),
                #nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                #nn.BatchNorm2d(ndf * 16),
                #nn.InstanceNorm2d(ndf * 16),
                #nn.LeakyReLU(0.2, inplace=True),
            ),
            # (B, nch_d*8, 8, 8) -> (B, 10, 3, 3)
        })

    def forward(self, x, img, label):
        for layer in self.layers.values():
            x = layer(x)
            print(x.shape)
        img = self.encode_img(img).view(-1,64 * 8, 4, 4)
        label = self.fc(label).view(-1,64 * 8,4,4)
        img_label = torch.concat((img,label),dim=1).view(-1,64 * 16 ,4,4)
        x = torch.concat((x,img_label),dim=1).view(-1,64*32,4,4)
        x = self.out(x)
        print(x.shape)
        return x.view(-1,1)


netG = Discriminator()
x = torch.ones((128,3,128,128))
label = torch.ones((128,10))
avg = nn.AvgPool2d(4)
img = avg(x)
print(img.shape)
#out = netG(x,img,label)


#custom_transform = transforms.Compose([
#                         transforms.Resize((128, 128)),
#                         transforms.RandomHorizontalFlip(),
#                         transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
#                     ])

#train_dataset = CelebaDataset(txt_path='label.txt',
#                              img_dir='.\\',
#                              transform=custom_transform)

#train_loader = DataLoader(dataset=train_dataset,
#                          batch_size=128,
#                          shuffle=True,
#                          num_workers=4)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#torch.manual_seed(0)



df = pd.read_csv('list_attr_celeba.txt', sep="\s+", skiprows=1, usecols=["Bald", "Blond_Hair", "Eyeglasses", "Heavy_Makeup", "Male", "Sideburns", "Smiling", "Wavy_Hair", "Wearing_Hat", "Young"])
#df.loc[df['Male'] == -1, 'Male'] = 0
#df.loc[df['Bald'] == -1, 'Bald'] = 0
#df.loc[df['Blond_Hair'] == -1, 'Blond_Hair'] = 0
#df.loc[df['Eyeglasses'] == -1, 'Eyeglasses'] = 0
#df.loc[df['Heavy_Makeup'] == -1, 'Heavy_Makeup'] = 0
#df.loc[df['Sideburns'] == -1, 'Sideburns'] = 0
#df.loc[df['Smiling'] == -1, 'Smiling'] = 0
#df.loc[df['Wavy_Hair'] == -1, 'Wavy_Hair'] = 0
#df.loc[df['Wearing_Hat'] == -1, 'Wearing_Hat'] = 0
#df.loc[df['Young'] == -1, 'Young'] = 0
df.to_csv('celeba_label.txt', sep=" ")

