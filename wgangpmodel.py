import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from transformers import BertTokenizer

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

def hw_flatten(x):
    """
        Flattens along the height and width of the tensor
    Args:
      x: Input tensor "NCHW"

    Returns:
        Flattened tensor
    """
    return torch.reshape(x, shape=(x.shape[0], x.shape[1], -1))


class Self_Attn(nn.Module):

    def __init__(self, in_channels, out_channels, sn=True):
        """
            Self-Attention Layer
        Args:
            in_channels : number of input channels
            out_channels : number of out channels
            sn : Flag for spectral normalization
        """
        super().__init__()
        self.out_channels = out_channels
        self.out_channels_sqrt = int(math.sqrt(out_channels))

        self.softmax = nn.Softmax(dim=-1)
        self.register_parameter(name='attention_multiplier', param=torch.nn.Parameter(torch.tensor(0.0)))

        self.f = nn.Conv2d(in_channels, self.out_channels_sqrt, kernel_size=(1, 1))
        self.g = nn.Conv2d(in_channels, self.out_channels_sqrt, kernel_size=(1, 1))
        self.h = nn.Conv2d(in_channels, self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        """ x -> NCHW """

        # TODO : seems like out_channels == in_channels should be satisfied
        s = torch.matmul(torch.transpose(hw_flatten(self.g(x)), 1, 2), hw_flatten(self.f(x)))

        beta = self.softmax(s)

        o = torch.matmul(beta, torch.transpose(hw_flatten(self.h(x)), 1, 2))
        o = torch.transpose(o, 1, 2)
        o = torch.reshape(o, x.shape)
        x = self.attention_multiplier * o + x
        return x


class block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride=(2, 2), dilation=(1, 1), pooling='avg',
                 padding='same', batch_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn0 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1)
        self.bn1 = nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=(1,1),dilation=dilation, padding=padding)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.upsample = nn.Upsample(scale_factor=stride, mode='nearest')
        self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel, stride=(1,1), padding=padding)
        self.upsample_0 = nn.Upsample(scale_factor=stride, mode='nearest')
        self.down_conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=(1,1), padding=padding)

    def forward(self, x):
        x_0 = x
        x = self.conv(x)
        x = self.bn0(x)
        x = self.act(x)
        x = self.upsample(x)
        x = self.down_conv(x)
        x_0 = self.upsample_0(x_0)
        x_0 = self.down_conv_0(x_0)
        out = x_0 + x
        print(out.shape)
        return out


class Generator(nn.Module):
    def __init__(self, zs_dim=100, gf_dim=10):
        super().__init__()

        self.z = zs_dim
        self.dim = gf_dim
        self.label_dim = 512
        self.linear = nn.Linear(self.z + self.label_dim, gf_dim * 512 * 16)

        self.block_stk = nn.Sequential(
            block(gf_dim * 16, gf_dim * 16),
            nn.AvgPool2d((2, 2), stride=2),
            block(gf_dim * 16, gf_dim * 8),
            nn.AvgPool2d((2, 2), stride=2),
            block(gf_dim * 8, gf_dim * 4),
            nn.AvgPool2d((2, 2), stride=2),
            block(gf_dim * 4, gf_dim * 2),
            nn.AvgPool2d((2, 2), stride=2),
            block(gf_dim * 2, gf_dim),
            nn.AvgPool2d((2, 2), stride=2),
            Self_Attn(gf_dim, gf_dim),
            block(gf_dim, gf_dim // 2),
            nn.AvgPool2d((2, 2), stride=2),
            nn.BatchNorm2d(gf_dim, eps=1e-05, momentum=0.1)
        )
        self.conv_stk = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(gf_dim, 1, kernel_size=(1, 1), stride=(1, 1)),
            nn.Tanh()
        )

    def forward(self, z, c):
        z = z.view(-1, self.z)
        c = c.view(-1, 512)
        x = torch.concat((z, c), dim=1).view(-1, 612)
        x = self.linear(x).view(-1, self.dim * 16, 512, 1)
        block_out = self.block_stk(x)
        print(block_out.shape)
        conv_out = self.conv_stk(block_out).squeeze()
        print(conv_out.shape)
        return conv_out


class block_(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=(3, 3), stride = (2, 2), dilation=(1, 1), padding='same'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel,dilation = dilation, padding=padding)
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.stride = stride
        self.down_conv = nn.Conv2d(out_channels, out_channels, kernel_size=kernel,padding=padding)
        self.avgPool2d = nn.AvgPool1d(kernel_size=2,stride=2)
        if stride[0] > 1 or stride[1] > 1 or self.in_channels != out_channels:
            self.down_conv_0 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel,padding=padding)
            self.avgPool2d_0 = nn.AvgPool1d(kernel_size=2,stride=2)

    def forward(self, x):
        x_0 = x
        x = self.conv(x)
        x = self.act(x)
        x = self.down_conv(x).squeeze()
        x = self.avgPool2d(x)
        if self.stride[0] > 1 or self.stride[1] > 1 or self.in_channels != self.out_channels:
            x_0 = self.down_conv_0(x_0).squeeze()
            x_0 = self.avgPool2d_0(x_0)
        out = x_0 + x
        return out


class Discriminator(nn.Module):
    def __init__(self, in_shape=1,df_dim=10,width=512, height=1):
        super().__init__()
        self.attn = Self_Attn(df_dim//2,df_dim//2)
        self.h0 = block_(in_shape, df_dim//2) # 12 * 12
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.h1 = block_(df_dim//2, df_dim ) # 6 * 6
        self.h2 = block_(df_dim, df_dim * 2) # 3 * 3
        self.h3 = block_(df_dim * 2, df_dim * 4)
        self.h4 = block_(df_dim * 4, df_dim * 8)
        self.h5 = block_(df_dim * 8, df_dim * 16) # 3 * 3
        self.h5_act = nn.LeakyReLU(negative_slope=0.2)
        self.conv = nn.Conv2d(df_dim * 16, df_dim * 16, kernel_size=(3,3), dilation=(1,1), padding="same")
        self.final = nn.Linear(df_dim * 8 * width * height, 1)
        self.sigmoid = nn.Sigmoid()  #yaogai

    def forward(self, x, label):
        input = torch.concat((x, label), dim=1).view(-1, 512 * 2, 1)
        x = torch.unsqueeze(input,dim=1)
        x = self.h0(x)
        #x = self.upsample(x)
        x = torch.unsqueeze(x, 3)
        x = self.attn(x)
        x = self.h1(x)
        x = self.upsample(x)
        x = torch.unsqueeze(x, 3)
        x = self.h2(x)
        x = self.upsample(x)
        x = torch.unsqueeze(x, 3)
        x = self.h3(x)
        x = self.upsample(x)
        x = torch.unsqueeze(x, 3)
        x = self.h4(x)
        x = self.upsample(x)
        x = torch.unsqueeze(x, 3)
        x = self.h5(x)
        x = self.upsample(x)
        x = torch.unsqueeze(x, 3)
        x = self.conv(x)
        #x = self.h5_act(x)
        x = torch.flatten(x, start_dim = 1)
        x = self.final(x)
        #x = self.sigmoid(x)
        return x.view(-1,1)

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


class Dataset(Dataset):
    """Custom Dataset for loading CelebA face images"""
    def __init__(self):
        df = pd.read_csv("data.csv",sep=",", index_col=0)
        self.x_1 = df["Organism"].values
        self.x_2 = df["Keywords"].values
        self.y = df["Sequence"].values


    def __getitem__(self, index):
        seq = self.y[index]
        label = self.x_1 +';' +self.x_2
        label = label[index]
        return seq, label

    def __len__(self):
        return self.y.shape[0]

trainset = Dataset()
dataloader = DataLoader(dataset=trainset,
                          batch_size=1,
                          shuffle=True,
                          num_workers=4)


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999),weight_decay=0.00001)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999),weight_decay=0.00001)



if __name__ == "__main__":
    num_epochs = 500
    latent_dim = 100
    lambda_gp = 10
    for epoch in range(num_epochs):
        for i, (seqs, labels) in enumerate(dataloader):
            batch_size = seqs.shape[0]
            seqs = tokenizer.encode(seqs, padding='max_length', max_length=512, add_special_tokens=True)
            labels = tokenizer(labels, padding='max_length', max_length=512, add_special_tokens=False)

            z = torch.randn(batch_size, latent_dim, device=device)
            discriminator.zero_grad()
            fake = generator(z, labels)
            real_validity = discriminator(seqs, labels)
            fake_validity = discriminator(fake, labels)
            gradient_penalty = compute_gradient_penalty(
                discriminator, seqs.data, fake.data,
                labels.data)

            d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty
            d_loss.backward()
            optimizer_D.step()
            generator.zero_grad()
            if i % 2 == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                fake = generator(z, labels)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                fake_validity = discriminator(fake, labels)
                g_loss = -torch.mean(fake_validity)

                g_loss.backward()
                optimizer_G.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [learning rate D/G: %f/%f]  [D loss: %f] [G loss: %f]"
                % (epoch, num_epochs, i, len(dataloader), optimizer_D.state_dict()['param_groups'][0]['lr'],
                   optimizer_G.state_dict()['param_groups'][0]['lr'], d_loss.item(), g_loss.item())
            )
