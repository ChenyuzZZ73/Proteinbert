import argparse
import os
import torch.autograd as autograd
import numpy as np
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F
from scipy import linalg
from PIL import Image
import pandas as pd


device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


outf = './result_50'

try:
    os.makedirs(outf)
except OSError:
    pass

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")


opt = parser.parse_args()
print(opt)

opt.img_shape = (opt.channels, opt.img_size, opt.img_size)

lambda_gp = 10


class InceptionV3(nn.Module):
    """Pretrained InceptionV3 network returning feature maps"""

    # Index of default block of inception to return,
    # corresponds to output of final average pooling
    DEFAULT_BLOCK_INDEX = 3

    # Maps feature dimensionality to their output blocks indices
    BLOCK_INDEX_BY_DIM = {
        64: 0,  # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        super(InceptionV3, self).__init__()
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3, \
            'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))
        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        """Get Inception feature maps
            Parameters
            ----------
            inp : torch.autograd.Variable
                Input tensor of shape Bx3xHxW. Values are expected to be in
                range (0, 1)
            Returns
            -------
            List of torch.autograd.Variable, corresponding to the selected output
            block, sorted ascending by index
        """
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                                  size=(299, 299),
                                  mode='bilinear',
                                  align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp

block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model = model.cuda()


def calculate_activation_statistics(images, model, batch_size=128, dims=2048,
                                    cuda=False):
    model.eval()
    act = np.empty((len(images), dims))

    if cuda:
        batch = images.cuda()
    else:
        batch = images
    pred = model(batch)[0]

    # If model output is not scalar, apply global spatial average pooling.
    # This happens if you choose a dimensionality not equal 2048.
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)


def calculate_fretchet(images_real, images_fake, model):
    mu_1, std_1 = calculate_activation_statistics(images_real, model, cuda=True)
    mu_2, std_2 = calculate_activation_statistics(images_fake, model, cuda=True)

    """get fretched distance"""
    fid_value = calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value






def weights_init(m):
    """
    Custom weights initialization as suggested in DCGAN article
    :param m: module
    :return:
    """
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d, nn.Linear]:
        nn.init.normal_(m.weight.data, 0.0, 0.02)

    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GeneratorI(nn.Module):
    def __init__(self,nz=100,ngf=64,nc=3):

        super().__init__()

        self.label = nn.Embedding(10,10)

        self.fc = nn.Sequential(
                #nn.ConvTranspose2d(nz + 10, ngf * 16, 5, 2, 1, bias=False),
                nn.Linear(in_features=nz + 10, out_features=ngf * 8 * 4 * 4),
                nn.BatchNorm1d(ngf * 8 * 4 * 4),
                nn.ReLU())

        self.layers = nn.ModuleDict({
            #'layer0': nn.Sequential(
                #nn.ConvTranspose2d(nz + 10, ngf * 16, 5, 2, 1, bias=False),
                #nn.Linear(in_features=nz + 10, out_features=ngf * 16 * 3 * 3)
                #nn.BatchNorm1d(ngf * 8),
                #nn.ReLU()

            #),  # (B, nz, 1, 1) -> (B, nch_g*8, 3, 3)
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2,padding=2, output_padding=1, bias=False),
                #nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1,bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU()
            ),  # (B, nch_g*8, 4, 4) -> (B, nch_g*4, 8, 8)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf * 2, 5, 2,padding=2, output_padding=1, bias=False),
                #nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU()
            ),  # (B, nch_g*4, 8, 8) -> (B, nch_g*2, 16, 16)

            'layer2': nn.Sequential(
                nn.ConvTranspose2d(ngf * 2, ngf, 5, 2,padding=2, output_padding=1, bias=False),
                #nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU()
            ),  # (B, nch_g*2, 16, 16) -> (B, nch_g, 32, 32)
            'layer3': nn.Sequential(
                #nn.ConvTranspose2d(ngf * 2, ngf, 5, 2,padding=2, output_padding=1, bias=False),
                nn.ConvTranspose2d(ngf, 3, 5, 2,padding=2, output_padding=1, bias=False),
                #nn.BatchNorm2d(ngf),
                #nn.ReLU()
                nn.Tanh()
            ),  # (B, nch_g, 32, 32) -> (B, nch, 64, 64)
        #'layer4': nn.Sequential(
        #    nn.ConvTranspose2d(ngf ,nc,  5, 2,padding=2, output_padding=1, bias=False),
        #    nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
        #    nn.Tanh()
        #)#（B,3,128,128)
        })

    def forward(self, noise , label):
        #label = self.label(label)
        z = torch.concat((noise,label), dim=1).view(-1,110)
        z = self.fc(z).view(-1,64*8,4,4)
        for layer in self.layers.values():
            z = layer(z)
            #print(z.shape)
        return z

class Generator(nn.Module):
    def __init__(self, GeneratorI,nz=100,ngf=64,nc=3):

        super().__init__()

        self.low_img = GeneratorI

        self.label = nn.Embedding(10, 10)

        self.encode_img = nn.Sequential(
                nn.Conv2d(nc, ngf, 5, 2, 2, bias=False),
                #nn.Linear(in_features=nz + 10, out_features=ngf * 16 * 4 * 4),
                nn.BatchNorm2d(ngf),
                nn.ReLU(),
                nn.Conv2d(ngf, ngf * 2, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(),
                nn.Conv2d(ngf * 2, ngf * 4, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(),
                nn.Conv2d(ngf * 4, ngf * 8, 5, 2, 2, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(),
        )#(ngf * 8 * 4 *4)

        self.fc = nn.Sequential(
                #nn.ConvTranspose2d(nz + 10, ngf * 16, 5, 2, 1, bias=False),
                nn.Linear(in_features=nz, out_features=ngf * 8* 4 * 4),
                nn.BatchNorm1d(ngf * 8 * 4 * 4),
                nn.ReLU()
        )

        self.layers = nn.ModuleDict({
            #'layer0': nn.Sequential(
                #nn.ConvTranspose2d(nz + 10, ngf * 16, 5, 2, 1, bias=False),
                #nn.Linear(in_features=nz + 10, out_features=ngf * 16 * 3 * 3)
                #nn.BatchNorm1d(ngf * 8),
                #nn.ReLU()

            #),  # (B, nz, 1, 1) -> (B, nch_g*8, 3, 3)
            'layer0': nn.Sequential(
                nn.ConvTranspose2d(ngf * 16, ngf * 8, 5, 2,padding=2, output_padding=1, bias=False),
                #nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1,bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU()
            ),  # (B, nch_g*8, 4, 4) -> (B, nch_g*4, 8, 8)
            'layer1': nn.Sequential(
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 5, 2,padding=2, output_padding=1, bias=False),
                #nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU()
            ),  # (B, nch_g*4, 8, 8) -> (B, nch_g*2, 16, 16)

            'layer2': nn.Sequential(
                nn.ConvTranspose2d(ngf * 4, ngf*2, 5, 2,padding=2, output_padding=1, bias=False),
                #nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf*2),
                nn.ReLU()
            ),  # (B, nch_g*2, 16, 16) -> (B, nch_g, 32, 32)
            'layer3': nn.Sequential(
                #nn.ConvTranspose2d(ngf*2, ngf, 5, 2,padding=2, output_padding=1, bias=False),
                nn.ConvTranspose2d(ngf*2, ngf, 5, 2,padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU()
                #nn.Tanh()
            ),  # (B, nch_g, 32, 32) -> (B, nch, 64, 64)
        'layer4': nn.Sequential(
            nn.ConvTranspose2d(ngf ,nc,  5, 2,padding=2, output_padding=1, bias=False),
           # nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )#（B,3,128,128)
        })

    def forward(self, noise, label):
        low = self.low_img(noise,label)
        #noise_label = torch.concat((noise,label), dim =1)
        noise = self.fc(noise).view(-1,64 * 8,4,4)
        img = self.encode_img(low).view(-1,64 * 8,4,4)
        z = torch.concat((noise,img), dim=1).view(-1,64 * 16,4,4)
        #z = self.fc(z).view(-1,128*16,4,4)
        for layer in self.layers.values():
            z = layer(z)
            #print(z.shape)
        return z,low


class DiscriminatorI(nn.Module):
    def __init__(self,nc=3,ndf=64):

        super().__init__()

        self.label = nn.Embedding(10, 10)

        self.fc = nn.Sequential(
            # nn.ConvTranspose2d(nz + 10, ngf * 16, 5, 2, 1, bias=False),
            nn.Linear(in_features=10, out_features= 10 * 128 * 128),
            nn.BatchNorm1d(10 * 128 * 128),
            nn.LeakyReLU())

        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.Conv2d(nc+10, ndf, 5, 2, 2, bias=False),
                #nn.Conv2d(nc + 10, ndf, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf),
                #nn.InstanceNorm2d(ndf),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (B, nch, 64, 64) -> (B, nch_d, 32, 32)
            'layer1': nn.Sequential(
                nn.Conv2d(ndf, ndf * 2, 5, 2, 2, bias=False),
                #nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf * 2),
                nn.InstanceNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (B, nch_d, 32, 32) -> (B, nch_d*2, 16, 16)
            'layer2': nn.Sequential(
                nn.Conv2d(ndf * 2, ndf * 4, 5, 2, 2, bias=False),
                #nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf * 4),
                nn.InstanceNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (B, nch_d*2, 16, 16) -> (B, nch_d*4, 8, 8)
            'layer3': nn.Sequential(
                nn.Conv2d(ndf * 4, ndf * 8, 5, 2, 2, bias=False),
                #nn.Conv2d(ndf * 4, 1, 4)
                #nn.BatchNorm2d(ndf * 8),
                nn.InstanceNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
            ),  # (B, nch_d*4, 8, 8) -> (B, nch_g*8, 4, 4)
            'layer4':   nn.Sequential(
                #nn.Conv2d(ndf*8, ndf * 16,5,2,2,bias=False),
                nn.Conv2d(ndf * 8, 1, 4)
                #nn.BatchNorm2d(ndf * 16),
                #nn.InstanceNorm2d(ndf * 16),
                #nn.LeakyReLU(0.2, inplace=True),
            ),
            # (B, nch_d*8, 4, 4) -> (B, 1, 1, 1)
        })

    def forward(self, x,  label):
        #label = self.label(label)
        label = label.view(-1,10,1,1)
        label = label.repeat(1,1,64,64)
        #print(label.shape)
        #label = self.fc(label).view(-1,10,128,128)
        x = torch.concat((x,label),dim=1)
        for layer in self.layers.values():
            x = layer(x)
            #print(x.shape)
        return x.view(-1,1)


class Discriminator(nn.Module):
    def __init__(self,nc=3,ndf=64):

        super().__init__()

        self.label = nn.Embedding(10, 10)

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
            nn.Linear(in_features= ndf*8+10, out_features=ndf * 8 * 4 * 4),
            nn.BatchNorm1d(ndf * 8 * 4 * 4),
            nn.ReLU())

        self.out = nn.Sequential(
            nn.Conv2d(ndf * 24 + 10, 1, 4))


        self.layers = nn.ModuleDict({
            'layer0': nn.Sequential(
                nn.Conv2d(nc+10, ndf, 5, 2, 2, bias=False),
                #nn.Conv2d(nc + 10, ndf, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(ndf),
                nn.InstanceNorm2d(ndf),
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
                nn.InstanceNorm2d(ndf * 16),
                nn.LeakyReLU(0.2, inplace=True),
            ),
           'layer5': nn.Sequential(
                nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False),
                # nn.BatchNorm2d(ndf * 16),
                #nn.InstanceNorm2d(ndf * 16),
                #nn.LeakyReLU(0.2, inplace=True),
           ),
        })

    def forward(self, x, label):
        #for layer in self.layers.values():
        #    x = layer(x)
            #print(x.shape)
        #img = self.encode_img(img).view(-1,64 * 8, 4, 4)
        #label = self.label(label)
        label = label.view(-1,10,1,1)
        label = label.repeat(1,1,128,128)
        #img_label = torch.concat((x,label),dim=1)
        x = torch.concat((x,label),dim=1)
        for layer in self.layers.values():
            x = layer(x)
            # print(x.shape)
        return x.view(-1,1)




#testset = dset.STL10(root='./STL10', download=True, split='test',
#                     transform=transforms.Compose([
#                         transforms.RandomResizedCrop(128),
#                         transforms.RandomHorizontalFlip(),
                         #transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
#                         transforms.ToTensor(),
#                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                     ]))
#dataset = trainset + testset

#dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
#                                         shuffle=True, num_workers=int(4))

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""
    def __init__(self, txt_path, img_dir, transform=None):
        df = pd.read_csv(txt_path, sep=" ", index_col=0)
        self.img_dir = img_dir
        self.txt_path = txt_path
        self.img_names = df.index.values
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

custom_transform = transforms.Compose([
                         transforms.Resize((128, 128)),
                         #transforms.RandomHorizontalFlip(),
                         #transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
                         transforms.ToTensor(),
                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                     ])

trainset = CelebaDataset(txt_path='celeba_label.txt',
                              img_dir='img_align_celeba/img_align_celeba',
                              transform=custom_transform)

dataloader = DataLoader(dataset=trainset,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          num_workers=4)


generatorI = GeneratorI().to(device)
generatorI.apply(weights_init)
discriminatorI = DiscriminatorI().to(device)
optimizer_GI = torch.optim.Adam(generatorI.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),weight_decay=0.00001)
optimizer_DI = torch.optim.Adam(discriminatorI.parameters(), lr=0.0004, betas=(opt.b1, opt.b2),weight_decay=0.00001)

generator = Generator(generatorI).to(device)
generator.apply(weights_init)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2),weight_decay=0.00001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0004, betas=(opt.b1, opt.b2),weight_decay=0.00001)
#scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_GI, step_size=50, gamma=0.5)
#scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_DI, step_size=50, gamma=0.5)


def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP.
       Warning: It doesn't compute the gradient w.r.t the labels, only w.r.t
       the interpolated real and fake samples, as in the WGAN GP paper.
    """
    # Random weight term for interpolation between real and fake samples
    alpha = np.random.random((real_samples.size(0), 1, 1, 1))
    alpha = torch.FloatTensor(alpha).to(device)
    #labels = LongTensor(labels)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = torch.FloatTensor(real_samples.shape[0], 1).fill_(1.0).to(device)
    fake.requires_grad = False
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    gradients = gradients[0].view(gradients[0].size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty



fixed_noise = torch.randn(70,opt.latent_dim, device=device)


fixed_label = [[-1, 1, -1, 1, -1, -1, 1, 1, -1, -1],  #金发老女人
               [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
               [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
               [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
               [-1, 1, -1, 1, -1, -1, 1, 1, -1, -1],
               [1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  #秃头男
               [1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  #秃头男
               [1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  # 秃头男
               [1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  # 秃头男
               [1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  # 秃头男
               [-1, -1, 1, -1, 1, -1, -1 ,-1, -1, 1],  #墨镜男
               [-1, -1, 1, -1, 1, -1, -1 ,-1, -1, 1],  #墨镜男
               [-1, -1, 1, -1, 1, -1, -1, -1, -1, 1],  # 墨镜男
               [-1, -1, 1, -1, 1, -1, -1, -1, -1, 1],  # 墨镜男
               [-1, -1, 1, -1, 1, -1, -1, -1, -1, 1],  # 墨镜男
               [-1, -1, -1, 1, -1, -1, 1, -1, -1, 1],  #heavymakeup
               [-1, -1, -1, 1, -1, -1, 1, -1, -1, 1],  # heavymakeup
               [-1, -1, -1, 1, -1, -1, 1, -1, -1, 1],  # heavymakeup
               [-1, -1, -1, 1, -1, -1, 1, -1, -1, 1],  # heavymakeup
               [-1, -1, -1, 1, -1, -1, 1, -1, -1, 1],  # heavymakeup
               [-1, -1, -1, -1, 1, -1 ,1, -1, -1, -1],  #oldman
               [-1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  # oldman
               [-1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  # oldman
               [-1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  # oldman
               [-1, -1, -1, -1, 1, -1, 1, -1, -1, -1],  # oldman
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],  # oldWOMANman
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],  # oldWOMANman
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],  # oldWOMANman
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],  # oldWOMANman
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, -1],  # oldWOMANman
               [-1, -1, -1, -1, 1, 1, -1, -1, -1, 1],  #络腮胡
               [-1, -1, -1, -1, 1, 1, -1, -1, -1, 1],  # 络腮胡
               [-1, -1, -1, -1, 1, 1, -1, -1, -1, 1],  # 络腮胡
               [-1, -1, -1, -1, 1, 1, -1, -1, -1, 1],  # 络腮胡
               [-1, -1, -1, -1, 1, 1, -1, -1, -1, 1],  # 络腮胡
               [-1, -1, -1, -1, -1 ,-1 ,1, -1, -1 ,1],  #smiling
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, 1],  # smiling
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, 1],  # smiling
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, 1],  # smiling
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, 1],  # smiling
               [-1, -1, -1, 1, -1, -1, -1, 1, -1 ,1],  #卷发
               [-1, -1, -1, 1, -1, -1, -1, 1, -1, 1],  # 卷发
               [-1, -1, -1, 1, -1, -1, -1, 1, -1, 1],  # 卷发
               [-1, -1, -1, 1, -1, -1, -1, 1, -1, 1],  # 卷发
               [-1, -1, -1, 1, -1, -1, -1, 1, -1, 1],  # 卷发
               [-1, -1, -1, -1, 1, -1, -1, -1, 1 ,1],  #戴帽子男人
               [-1, -1, -1, -1, 1, -1, -1, -1, 1 ,1],  #戴帽子男人
               [-1, -1, -1, -1, 1, -1, -1, -1, 1, 1],  # 戴帽子男人
               [-1, -1, -1, -1, 1, -1, -1, -1, 1, 1],  # 戴帽子男人
               [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1],  # 戴帽子n人
               [-1, -1, -1, 1, -1, -1, -1, -1, -1, 1],  #浓妆
               [-1, -1, -1, 1, -1, -1, -1, -1, -1, 1],  # 浓妆
               [-1, -1, -1, 1, -1, -1, 1, -1, -1, 1],  # 浓妆
               [-1, -1, -1, 1, -1, -1, 1, -1, -1, 1],  # 浓妆
               [-1, -1, -1, 1, -1, -1, 1, -1, -1, 1],  # 浓妆
               [-1, 1, -1, -1, 1, -1,-1 ,-1 ,-1 ,-1],  #金发男
               [-1, 1, -1, -1, 1, -1, -1, -1, -1, -1],  # 金发男
               [-1, 1, -1, -1, 1, -1, 1, -1, -1, -1],  # 金发男
               [-1, 1, -1, -1, 1, -1, 1, -1, -1, -1],  # 金发男
               [-1, 1, -1, -1, 1, -1, 1, -1, -1, -1],  # 金发男
               [1, -1, -1, -1, 1, 1, 1, -1, -1, 1],  #胡子秃头男
               [1, -1, -1, -1, 1, 1, 1, -1, -1, 1],  #胡子秃头男
               [1, -1, -1, -1, 1, 1, 1, -1, -1, 1],  #胡子秃头男
               [1, -1, -1, -1, 1, 1, 1, -1, -1, 1],  #胡子秃头男
               [1, -1, -1, -1, 1, 1, 1, -1, -1, 1],  #胡子秃头男
               [-1, -1, -1, -1, -1 ,-1 ,1, -1, -1, 1],  #淡妆笑女
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, 1],  # 淡妆笑女
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, 1],  # 淡妆笑女
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, 1],  # 淡妆笑女
               [-1, -1, -1, -1, -1, -1, 1, -1, -1, 1]]  # 淡妆笑女

fixed_label = torch.LongTensor(fixed_label).to(device)

if __name__ == "__main__":
    batches_done = 0
    for epoch in range(opt.n_epochs):
        for i, (imgs, labels) in enumerate(dataloader):
            batch_size = imgs.shape[0]
            avg = nn.AvgPool2d(2)
            # Move to GPU if necessary
            real_imgs = torch.FloatTensor(imgs).to(device)
            labels = torch.LongTensor(labels).to(device)
            low_imgs = avg(real_imgs)

            z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)

            optimizer_DI.zero_grad()
                # Generate a batch of images
            fake_low_imgs = generatorI(z, labels)
                # Real images
            real_validityI = discriminatorI(low_imgs, labels)
                # Fake images
            fake_validityI = discriminatorI(fake_low_imgs, labels)
                # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                    discriminatorI, low_imgs.data, fake_low_imgs.data,
                    labels.data)
            d_lossI = -torch.mean(real_validityI) + torch.mean(fake_validityI) + lambda_gp * gradient_penalty

            d_lossI.backward()
            optimizer_DI.step()

            optimizer_GI.zero_grad()

            if i % opt.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                fake_low_imgs = generatorI(z, labels)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                fake_validityI = discriminatorI(fake_low_imgs, labels)
                g_lossI = -torch.mean(fake_validityI)

                g_lossI.backward()
                optimizer_GI.step()

            generator.low_img = generatorI
                # ---------------------
                #  Train Discriminator
                # ---------------------
            optimizer_DI.zero_grad()
            optimizer_D.zero_grad()

                # Sample noise and labels as generator input
                #z = torch.randn(imgs.shape[0], opt.latent_dim, device=device)
                # Generate a batch of images
            fake_imgs, _ = generator(z, labels)

                # Real images
            real_validity = discriminator(real_imgs,  labels)
                # Fake images
            fake_validity = discriminator(fake_imgs,  labels)
                # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, real_imgs.data, fake_imgs.data,
                labels.data)
                # Adversarial loss
            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

            d_loss.backward()
            optimizer_D.step()
            optimizer_DI.step()

            optimizer_G.zero_grad()
            optimizer_GI.zero_grad()


                # Train the generator every n_critic steps
            if i % opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                fake_imgs,_ = generator(z, labels)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                fake_validity = discriminator(fake_imgs, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()
                optimizer_GI.step()

            fretchet_dist = calculate_fretchet(imgs, fake_imgs, model)

            print(
                "[Epoch %d/%d] [Batch %d/%d] [learning rate D/G: %f/%f]  [D loss: %f] [G loss: %f] [FID: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), optimizer_D.state_dict()['param_groups'][0]['lr'],
                   optimizer_G.state_dict()['param_groups'][0]['lr'], d_loss.item(), g_loss.item(),fretchet_dist)
            )

        fake_noise = fixed_noise
        fake_label = fixed_label


        fake, low = generator(fake_noise, fake_label)
        vutils.save_image(fake.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),
                          normalize=True, nrow=10)
        vutils.save_image(low.detach(), '{}/low_samples_epoch_{:03d}.png'.format(outf, epoch + 1),
                          normalize=True, nrow=10)

        torch.save(generatorI.state_dict(),'GeneratorI.pt')
        torch.save(generator.state_dict(), 'Generator.pt')

