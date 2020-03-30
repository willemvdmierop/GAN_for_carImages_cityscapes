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
import Model_ResNet_GAN
# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline



device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')


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

#################################### Train Parameters for Dis and GEN ################################
wd = os.getcwd()
batch_size, image_size = 256, [64, 64]
assert len(image_size) == 2
# Epochs
num_epochs = 4001
# number of channels
nc = 3
# latent space (z) size: G input
latentVect = 100
# Feature vector of Discriminator
FeaDis = 64
# Feature vector of generator
FeaGen = 64
#path_img = os.path.join(wd, "cars3_green")
# this is just for now, use path above for server training
path_img = "/Users/willemvandemierop/Google Drive/DL Classification (705)/v_03_with_carimages/cars3_green"
for filename in sorted(os.listdir(path_img)):
    if filename == '.DS_Store':
        os.remove(path_img + "/" + filename)

# ====================================== dataset and dataloader ====================================== #
dataset_pars = {'img_size': image_size, 'img_data': path_img}
obs = Cars(**dataset_pars)
dataloader_pars = {'batch_size': batch_size, 'shuffle': True}
dataloader = data.DataLoader(obs, **dataloader_pars)

# =============================== Instantiate models and put them on CUDA ============================ #
g_pars = {'z_dim': latentVect, 'in_planes': FeaGen, 'channels': nc}
d_pars = {'in_planes': FeaDis, 'channels': nc}

d = Model_ResNet_GAN.ResNet_Discriminator(Model_ResNet_GAN.BasicBlock,[2,2,2,2],**d_pars)
# weights_d = torch.load(os.path.join(wd, "dis_gr_ResN_LR1e-4_WD1e-3_Batch256_2000.pth"))
# d.load_state_dict(weights_d)
d = d.to(device)
print(d)

total_d = 0
for _n, _par in d.state_dict().items():
    total_d += _par.numel()

print("parameters discriminator", total_d)

g = Model_ResNet_GAN.ResNet_Generator(Model_ResNet_GAN.Generator_BasicBlock,[2,2,2,2], **g_pars)
# weights_g = torch.load(os.path.join(wd, "gen_gr_ResN_LR1e-4_WD1e-3_Batch256_2000.pth"))
# g.load_state_dict(weights_g)
g = g.to(device)
print(g)

total_g = 0
for _n, _par in g.state_dict().items():
    total_g += _par.numel()

print("parameters generator", total_g)


#
print('# ' + '=' *45 + ' Training ' + '=' * 45 + ' #')
# ============================================= Training ============================================= #
# create labels
real_label = 1
generated_label = 0

# optimizers
lrate = 1e-4
optimizer_pars = {'lr': lrate, 'weight_decay': 1e-3}
optimizerD = optim.Adam(d.parameters(), **optimizer_pars)
optimizerG = optim.Adam(g.parameters(), **optimizer_pars)

if not os.path.exists(wd + '/gen_images_green_ResNet'):
    os.mkdir(wd + '/gen_images_green_ResNet')

tb = SummaryWriter(comment="ResNet_GAN_LR_1e-4_BATCH_256_WD_1e-3")
loss = BCELoss()
loss = loss.to(device)

img_list = []
# main train loop
for e in range(2000, num_epochs):
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

    if e % 250 == 0:
        torch.save(g.state_dict(), os.path.join(wd, "gen_gr_ResN_LR1e-4_WD1e-3_256_" + str(e) + ".pth"))
        torch.save(d.state_dict(), os.path.join(wd, "dis_gr_ResN_LR1e-4_WD1e-3_256_"+ str(e) + ".pth"))
        print("saved intermediate weights")
        weights = torch.load(os.path.join(wd,"gen_gr_ResN_LR1e-4_WD1e-3_256_" + str(e) + ".pth" ))
        args = {'latentVect': 100, 'FeaGen': 128, 'nc': 3}
        model = Model_ResNet_GAN.ResNet_Generator(Model_ResNet_GAN.Generator_BasicBlock,[2,2,2,2], **g_pars)
        model.load_state_dict(weights)
        z = torch.randn(1, 100, 1, 1)
        out = model(z)
        t_ = transforms.Normalize(mean=[-0.485, -0.450, -0.407], std=[1, 1, 1])
        out = out.detach().clone().squeeze_(0)
        out = t_(out).numpy().transpose(1, 2, 0)
        plt.imshow(out)
        filename = wd + "/gen_images_green_ResNet/" + "img_gen_gr_ResNet_" + str(e) + ".png"
        plt.savefig(filename)


tb.close()
torch.save(g.state_dict(), os.path.join(wd, "gen_gr_ResN_LR1e-4_WD1e-3_256_" + str(e) + ".pth"))
torch.save(d.state_dict(), os.path.join(wd, "dis_gr_ResN_LR1e-4_WD1e-3_256_" + str(e) + ".pth"))
print("finished training")


















