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
from torch.utils.tensorboard import SummaryWriter

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


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
        self.latentVect = kwargs['latentVect']
        self.FeaGen = kwargs['FeaGen']
        self.nc = kwargs['nc']
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(in_channels=self.latentVect, out_channels=self.FeaGen * 8, kernel_size=4, stride=1,
                               padding=0,
                               bias=False),
            nn.BatchNorm2d(self.FeaGen * 8),
            nn.ReLU(True),
            # state size. (FeaGen*8) x 4 x 4
            nn.ConvTranspose2d(self.FeaGen * 8, self.FeaGen * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.FeaGen * 4),
            nn.ReLU(True),
            # state size. (FeaGen*4) x 8 x 8
            nn.ConvTranspose2d(self.FeaGen * 4, self.FeaGen * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.FeaGen * 2),
            nn.ReLU(True),
            # state size. (FeaGen*2) x 16 x 16
            nn.ConvTranspose2d(self.FeaGen * 2, self.FeaGen, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.FeaGen),
            nn.ReLU(True),
            # state size. (FeaGen) x 32 x 32
            nn.ConvTranspose2d(self.FeaGen, self.nc, 4, 2, 1, bias=False),
            nn.Tanh()

            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


# Discriminator takes an 'image': object dimensionality batch_size x 3 x H x W
class Discriminator(nn.Module):
    def __init__(self, **kwargs):
        super(Discriminator, self).__init__()
        self.FeaDis = kwargs['FeaDis']
        self.nc = kwargs['nc']
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(in_channels=self.nc, out_channels=self.FeaDis, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (FeaDis) x 32 x 32
            nn.Conv2d(self.FeaDis, self.FeaDis * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.FeaDis * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (FeaDis*2) x 16 x 16
            nn.Conv2d(self.FeaDis * 2, self.FeaDis * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.FeaDis * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (FeaDis*4) x 8 x 8
            nn.Conv2d(self.FeaDis * 4, self.FeaDis * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.FeaDis * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (FeaDis*8) x 4 x 4
            nn.Conv2d(self.FeaDis * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 11 * 11, 1),
            # nn.Sigmoid()
        )

    def forward(self, input):
        features = self.main(input)
        # features = features.view(-1, features.size()[1]*features.size()[2]*features.size()[3])
        # out=self.classifier(features)
        return features


# ========================================= Training Parameters ======================================== #
wd = os.getcwd()
batch_size, image_size = 256, [64, 64]
batch_size_str = str(batch_size)
assert len(image_size) == 2
# Epochs
num_epochs = 4501
# number of channels
nc = 3
# latent space (z) size: G input
latentVect = 100
# Feature vector of Discriminator
FeaDis = 64
# Feature vector of generator
FeaGen = 64
# optimizers
lrate = 1e-4
optimizer_pars = {'lr': lrate, 'weight_decay': 1e-3}
w_decay_str = '001'
lrate_str = '0001'

# ====================================== dataset and dataloader ====================================== #

path_img = os.path.join(wd, "cars3_green")
# path_img = "/Users/willemvandemierop/Google Drive/DL Classification (705)/v_03_with_carimages/cars3"
for filename in sorted(os.listdir(path_img)):
    if filename == '.DS_Store':
        os.remove(path_img + "/" + filename)

dataset_pars = {'img_size': image_size, 'img_data': path_img}
obs = Cars(**dataset_pars)
dataloader_pars = {'batch_size': batch_size, 'shuffle': True}
dataloader = data.DataLoader(obs, **dataloader_pars)

# =============================== Instantiate models and put them on CUDA ============================ #

dirname = 'model_DCGAN_batch' + str(batch_size) + "_wd" + w_decay_str + "_lr" + lrate_str
if not os.path.exists(os.path.join(wd, dirname)): os.mkdir(os.path.join(wd, dirname))

g_pars = {'latentVect': latentVect, 'FeaGen': FeaGen, 'nc': nc}
d_pars = {'FeaDis': FeaDis, 'nc': nc}
# initialize the models
g = Generator(**g_pars)
d = Discriminator(**d_pars)
g = g.to(device)
d = d.to(device)

# =============================================== Optimizers ========================================= #
# create labels
real_label = 1
generated_label = 0

optimizerD = optim.Adam(d.parameters(), **optimizer_pars)
optimizerG = optim.Adam(g.parameters(), **optimizer_pars)

if not os.path.exists(wd + '/gen_images_gr_DCGAN'):
    os.mkdir(wd + '/gen_images_gr_DCGAN')

# =========================================== Pretrained Loading ===================================== #
epochs = 0
folder_name = os.path.join(wd, dirname)
if os.path.exists(os.path.join(folder_name, 'checkpoint.pth')):
    print("loading pretrained optimizers")
    checkpoint = torch.load(os.path.join(dirname, 'checkpoint.pth'))
    optimizerD.load_state_dict(checkpoint['optimizer_state_dict_D'])
    optimizerG.load_state_dict(checkpoint['optimizer_state_dict_G'])
    epochs = checkpoint['epoch'] + 1
    try:

        weights_g = torch.load(os.path.join(folder_name, "gen_gr_DC_batch_" + batch_size_str + "_wd" + w_decay_str +
                                            "_lr" + lrate_str + "_e" + str(epochs - 1) + ".pth"))
        g.load_state_dict(weights_g)
        print("Loaded pretrained generator")
    except:
        raise FileNotFoundError("could not load Generator")
    try:
        weights_g = torch.load(os.path.join(folder_name, "gen_gr_DC_batch_" + batch_size_str + "_wd" + w_decay_str +
                                            "_lr" + lrate_str + "_e" + str(epochs - 1) + ".pth"))
        g.load_state_dict(weights_g)
        print("Loaded pretrained generator")
    except:
        raise FileNotFoundError("could not load Generator")


# =========================================== Print parameters of models ===================================== #
'''
print(g)
total_g = 0
for _n, _par in g.state_dict().items():
    total_g += _par.numel()

print("parameters generator", total_g)
print(d)
total_d = 0
for _n, _par in d.state_dict().items():
    total_d += _par.numel()
print("parameters discriminator", total_d)
'''



# ================================================= Training ================================================== #
# loss function
tb = SummaryWriter(comment="DC_GAN_batch" + str(batch_size) + "_wd" + w_decay_str + "_lr" + lrate_str)
loss = BCELoss()
loss = loss.to(device)
# main train loop
if epochs >= num_epochs:
    raise SyntaxError("we have already trained for this amount of epochs")
for e in range(epochs, num_epochs):
    for id, data in dataloader:
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
        z = torch.randn(batch_size, latentVect, 1, 1, device=device)
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

    if e % 100 == 0:
        ## let's save the optimizers
        torch.save({'epoch': e, 'optimizer_state_dict_D': optimizerD.state_dict(),
                    "optimizer_state_dict_G": optimizerG.state_dict()}, os.path.join(folder_name, 'checkpoint.pth'))
        ## let's save the weights of the models
        torch.save(g.state_dict(), os.path.join(folder_name, "gen_gr_DC_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
        torch.save(d.state_dict(), os.path.join(folder_name, "dis_gr_DC_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
        print("saved intermediate weights")
        ## let's load the model to generate images.
        weights = torch.load(os.path.join(folder_name, "gen_gr_DC_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
        args = {'latentVect': 100, 'FeaGen': 128, 'nc': 3}
        model = Generator(**g_pars)
        model.load_state_dict(weights)
        for i in range(5):
            if not os.path.exists(wd + "/gen_images_gr_DCGAN/" + "hallucinated_" + str(e)):
                os.mkdir(wd + "/gen_images_gr_DCGAN/" + "hallucinated_" + str(e))
            z = torch.randn(1, 100, 1, 1)
            out = model(z)
            t_ = transforms.Normalize(mean=[-0.485, -0.450, -0.407], std=[1, 1, 1])
            out = out.detach().clone().squeeze_(0)
            out = t_(out).numpy().transpose(1, 2, 0)
            plt.imshow(out)
            filename = wd + "/gen_images_gr_DCGAN/" + "hallucinated_" + str(e) + "/generated_" + str(i) + ".png"
            plt.savefig(filename)

tb.close()
## let's save the optimizers
torch.save({'epoch': e, 'optimizer_state_dict_D': optimizerD.state_dict(),
            "optimizer_state_dict_G": optimizerG.state_dict()}, os.path.join(folder_name, 'checkpoint.pth'))
torch.save(g.state_dict(), os.path.join(folder_name, "gen_gr_DC_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
torch.save(d.state_dict(), os.path.join(folder_name, "dis_gr_DC_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
print("finished training")
