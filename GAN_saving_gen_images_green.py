# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torch import optim
from torch.nn import BCELoss
from torch.utils import data
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

class Cars(data.Dataset):
    def __init__(self, **kwargs):
        self.img_data = kwargs['img_data']
        self.img_size = kwargs['img_size']
        self.imgs = os.listdir(self.img_data)

    def transform_img(self, img, img_size):
        h_, w_ = img_size[0], img_size[1]
        im_size = tuple([h_, w_])
        t_ = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(im_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.457, 0.407],
                                 std=[1, 1, 1])
        ])
        img = t_(img)
        return img

    def load_img(self, idx):
        im = np.array(PILImage.open(os.path.join(self.img_data, self.imgs[idx])))
        im = self.transform_img(im, self.img_size)
        return im

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        X = self.load_img(idx)
        return idx, X


class Generator(nn.Module):
    def __init__(self, **kwargs):
        super(Generator, self).__init__()
        self.nz = kwargs['nz']
        self.ngf = kwargs['ngf']
        self.nc = kwargs['nc']
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.nz, out_channels=self.ngf * 8, kernel_size=8, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(self.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(self.ngf * 4, self.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(self.ngf * 2, self.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(self.ngf, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator takes an 'image': object dimensionality batch_size x 3 x H x W
class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.ndf = kwargs['ndf']
        self.nc = kwargs['nc']
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(self.ndf * 2, self.ndf * 4,kernel_size= 4,stride= 2,padding= 1, bias=False),
            nn.BatchNorm2d(self.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(self.ndf * 4, self.ndf * 8, kernel_size=4,stride= 2,padding= 1, bias=False),
            nn.BatchNorm2d(self.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(self.ndf * 8, 1, kernel_size=7,stride= 2,padding= 0, bias=False),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 11 * 11, 1),
            # nn.Sigmoid()
        )

    def forward(self, input):
        features = self.main(input)
        # features = features.view(-1, features.size()[1]*features.size()[2]*features.size()[3])
        # out=self.classifier(features)
        return features


batch_size = 256
ngf = 128
image_size = [128, 128]
assert len(image_size) == 2
wd = os.getcwd()
# wd = '/cars5'
# number of channels
nc = 3
# latent space (z) size: G input
nz = 100
# 
ndf = 128
num_epochs = 1
#path_img = os.path.join(wd, "cars3_green")
# this is just for now, use path above for server training
path_img = "/Users/willemvandemierop/Documents/Master AI/Pycharm/DL Classification/Coursework/v_03_with_carimages/cars4"
for filename in sorted(os.listdir(path_img)):
    if filename == '.DS_Store':
        os.remove(path_img + "/" + filename)

# dataset and dataloader
dataset_pars = {'img_size': image_size, 'img_data': path_img}
obs = Cars(**dataset_pars)
dataloader_pars = {'batch_size': batch_size, 'shuffle': True}
dataloader = data.DataLoader(obs, **dataloader_pars)
# get the right device

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

print(device)
# instantiate models and put them on CUDA
g_pars = {'nz': nz, 'ngf': ngf, 'nc': nc}
d_pars = {'ndf': ndf, 'nc': nc}

# models
print("\nloading weights from previously trained discriminator and generator\n")
g = Generator(**g_pars)
d = Discriminator(**d_pars)
g = g.to(device)
d = d.to(device)

print(g)
print(d)

# create labels
real_label = 1
generated_label = 0

# optimizers
lrate = 1e-4
optimizer_pars = {'lr': lrate, 'weight_decay': 1e-3}
optimizerD = optim.Adam(d.parameters(), **optimizer_pars)
optimizerG = optim.Adam(g.parameters(), **optimizer_pars)

if not os.path.exists(wd + '/generated_images_green'):
    os.mkdir(wd + '/generated_images_green')

tb = SummaryWriter()
loss = BCELoss()
loss = loss.to(device)

img_list = []
# main train loop
for e in range(num_epochs):
    for id, data in dataloader:
        # print(id)
        # first, train the discriminator
        d.zero_grad()
        g.zero_grad()
        data = data.to(device)
        batch_size = data.size()[0]
        # labels: all 1s
        labels_t = torch.ones(batch_size).unsqueeze_(0)
        labels_t = labels_t.to(device)
        # get the prediction of the D model
        # D(x)
        predict_d = d(data).view(1, -1)
        # loss from real data
        loss_t = loss(predict_d, labels_t)
        loss_t.backward()
        # generator input and output
        z = torch.randn(batch_size, nz, 1, 1, device=device)
        h = g(z)
        # all labels are 0s
        labels_g = torch.zeros(batch_size).unsqueeze_(0)
        labels_g = labels_g.to(device)
        # D(1-G(z))
        predict_g = d(h.detach()).view(1, -1)
        # loss from generated data
        loss_g = loss(predict_g, labels_g)
        loss_g.backward()
        total_loss = loss_t + loss_g
        # update discriminator weights
        optimizerD.step()
        # D(G(z))
        g.zero_grad()
        labels_g_real = torch.ones(batch_size).unsqueeze_(0)
        labels_g_real = labels_g_real.to(device)
        # 
        o = d(h).view(1, -1)
        loss_g_real = loss(o, labels_g_real)
        # update generator weights
        loss_g_real.backward()
        optimizerG.step()

        tb.add_scalar('Discriminator Loss w.r.t. Real Data (D(x))', loss_t, e)
        tb.add_scalar('Discriminator Loss w.r.t. Generated Data (D(1-G(z)))', loss_g, e)
        tb.add_scalar('Total Loss', total_loss, e)
    """
    if e % 500 == 0:
        torch.save(g.state_dict(), os.path.join(wd, "car_generator_GAN_intermediate_green" + str(e+1) + ".pth"))
        torch.save(d.state_dict(), os.path.join(wd, "car_discriminator_GAN_intermediate_green"+ str(e+1) + ".pth"))
        print("saved intermediate weights")
        weights = torch.load(os.path.join(wd,"car_generator_GAN_intermediate_green" + str(e+1) + ".pth" ))
        args = {'nz': 100, 'ngf': 64, 'nc': 3}
        model = Generator(**args)
        model.load_state_dict(weights)
        z = torch.randn(1, 100, 1, 1)
        out = model(z)
        t_ = transforms.Normalize(mean=[-0.485, -0.450, -0.407], std=[1, 1, 1])
        out = out.detach().clone().squeeze_(0)
        out = t_(out).numpy().transpose(1, 2, 0)
        plt.imshow(out)
        filename = wd + "/generated_images_green/" + "img_generator_green" + str(e+1) + ".png"
        plt.savefig(filename)
        
    """

tb.close()
#torch.save(g.state_dict(), os.path.join(wd, "car_generator_GAN_green" + str(e+1) + ".pth"))
#torch.save(d.state_dict(), os.path.join(wd, "car_discriminator_GAN_green" + str(e+1) + ".pth"))
print("finished training")
