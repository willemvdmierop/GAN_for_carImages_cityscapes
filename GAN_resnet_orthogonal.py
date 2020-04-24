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
import Model_ResNet_GAN_ortho as Model_ResNet_GAN

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

# ================================================== Dataset ============================================== #
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


# ========================================= Training Parameters ======================================== #

#################################
### Training Hyper-Parameters ###
#################################

wd = os.getcwd()
batch_size, image_size = 256, [64, 64]
batch_size_str = str(batch_size)
assert len(image_size) == 2
# Epochs
num_epochs = 1
# number of channels
nc = 3
# latent space (z) size: G input
latentVect = 100
# Feature vector of Discriminator
FeaDis = 64
# Feature vector of generator
FeaGen = 64
#### chose your Resnet:##
ResN18 = True ###########
ResN34 = False ##########
Gradient_clip_on = True #
max_grad_norm = 1.0 #####
#########################
# optimizers
lrate = 1e-4
lrate_str = '0001'
w_decay = 1e-3
w_decay_str = '001'
optimizer_pars = {'lr': lrate, 'weight_decay': 1e-3}
if ResN18: ResNet_str = 'ResNet18'
if ResN34: ResNet_str = 'ResNet34'

dirname = 'model_' + ResNet_str + '_batch' + str(batch_size) + "_wd" + w_decay_str + "_lr" + lrate_str
if not os.path.exists(os.path.join(wd, dirname)): os.mkdir(os.path.join(wd, dirname))

# ============================================= Dataset path ========================================= #
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

# ResNet18: parameters discriminator 11183318
if ResN18:
    d = Model_ResNet_GAN.ResNet_Discriminator(Model_ResNet_GAN.BasicBlock,[2,2,2,2],**d_pars)
# ResNet34: parameters discriminator 21298918
if ResN34:
    d = Model_ResNet_GAN.ResNet_Discriminator(Model_ResNet_GAN.BasicBlock, [3, 4, 6, 3], **d_pars)
# weights_d = torch.load(os.path.join(wd, "dis_gr_ResN_LR1e-4_WD1e-3_Batch256_2000.pth"))
# d.load_state_dict(weights_d)
d = d.to(device)

# ResNet18: parameters generator 14689046
if ResN18:
    g = Model_ResNet_GAN.ResNet_Generator(Model_ResNet_GAN.Generator_BasicBlock,[2,2,2,2], **g_pars)
# ResNet34: parameters generator 22958884
if ResN34:
    g = Model_ResNet_GAN.ResNet_Generator(Model_ResNet_GAN.Generator_BasicBlock, [3, 4, 6, 3], **g_pars)
# weights_g = torch.load(os.path.join(wd, "gen_gr_ResN_LR1e-4_WD1e-3_Batch256_2000.pth"))
# g.load_state_dict(weights_g)
g = g.to(device)
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



# orthogonal parameter regularization. Source: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/utils.py
def ortho_reg(model, strength=1e-4, blacklist=[]):
    with torch.no_grad():
        for param in model.parameters():
            # Regularize only parameters with at least 2 dim or whitelisted
            if len(param.shape) < 2 or any([param is item for item in blacklist]):
                continue
            # take all parameters to matrix shape, i.e. 2 dim
            w = param.view(param.shape[0], -1)
            # apply orthogonal regularization
            grad = (2 * torch.mm(torch.mm(w, w.t())
                                 * (1. - torch.eye(w.shape[0], device=w.device)), w))
            # update regularized parameters
            param.grad.data += strength * grad.view(param.shape)


# =============================================== Optimizers ========================================= #
# create labels
real_label = 1
generated_label = 0

optimizerD = optim.Adam(d.parameters(), **optimizer_pars)
optimizerG = optim.Adam(g.parameters(), **optimizer_pars)

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
        print("Loading pretrained generator")
        weights_g = torch.load(os.path.join(folder_name, "gen_gr_ResN_batch_" + batch_size_str + "_wd"
                                            + w_decay_str + "_lr" + lrate_str + "_e" + str(epochs -1) + ".pth"))
        g.load_state_dict(weights_g)
    except:
        raise FileNotFoundError("could not load Generator")
    try:
        print("Loading pretrained discriminator")
        weights_d = torch.load(os.path.join(folder_name, "dis_gr_ResN_batch_" + batch_size_str + "_wd"
                                            + w_decay_str + "_lr" + lrate_str + "_e" + str(epochs - 1) + ".pth"))
        d.load_state_dict(weights_d)
    except:
        raise FileNotFoundError("could not load Discriminator")

# ============================================= Training ============================================= #
if not os.path.exists(wd + '/gen_images_green_' + ResNet_str):
    os.mkdir(wd + '/gen_images_green_' + ResNet_str)

tb = SummaryWriter(comment=ResNet_str + "_GAN_Orthogonal_batch" + str(batch_size) + "_wd" + w_decay_str + "_lr" + lrate_str + "epochs_" + str(num_epochs))
loss = BCELoss()
loss = loss.to(device)
img_list = []
# main train loop
if epochs >= num_epochs:
    raise SyntaxError("we have already trained for this amount of epochs")
for e in range(epochs, num_epochs):
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
        # apply orthogonal regularization to discriminator parameters
        ortho_reg(d)
        # update discriminator parameters
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
        # apply orthogonal regularization to generator parameters
        ortho_reg(g)
        # update generator parameters
        optimizerG.step()
        if Gradient_clip_on:
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(g.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(d.parameters(), max_grad_norm)

        tb.add_scalar('Discriminator Loss w.r.t. Real Data (D(x))', loss_t, e)
        tb.add_scalar('Discriminator Loss w.r.t. Generated Data (D(1-G(z)))', loss_g, e)
        tb.add_scalar('Total Loss', total_loss, e)

    if e % 100 == 0:
        ## let's save the optimizers
        torch.save({'epoch': e, 'optimizer_state_dict_D': optimizerD.state_dict(),
                    "optimizer_state_dict_G": optimizerG.state_dict()}, os.path.join(folder_name,'checkpoint.pth'))
        ## let's save the weights
        torch.save(g.state_dict(), os.path.join(folder_name, "gen_gr_ResN_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
        torch.save(d.state_dict(), os.path.join(folder_name, "dis_gr_ResN_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
        print("saved intermediate weights")
        ## let's load the model to generate images.
        weights = torch.load(os.path.join(folder_name, "gen_gr_ResN_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
        args = {'latentVect': 100, 'FeaGen': 128, 'nc': 3}
        if ResN18:
            model = Model_ResNet_GAN.ResNet_Generator(Model_ResNet_GAN.Generator_BasicBlock, [2, 2, 2, 2], **g_pars)
        if ResN34:
            model = Model_ResNet_GAN.ResNet_Generator(Model_ResNet_GAN.Generator_BasicBlock, [3, 4, 6, 3], **g_pars)
        model.load_state_dict(weights)
        for i in range(5):
            if not os.path.exists(wd + '/gen_images_green_' + ResNet_str + "/hallucinated_" + str(e)):
                os.mkdir(wd + '/gen_images_green_' + ResNet_str + "/hallucinated_" + str(e))
            z = torch.randn(1, 100, 1, 1)
            out = model(z)
            t_ = transforms.Normalize(mean=[-0.485, -0.450, -0.407], std=[1, 1, 1])
            out = out.detach().clone().squeeze_(0)
            out = t_(out).numpy().transpose(1, 2, 0)
            plt.imshow(out)
            filename = wd + "/gen_images_green_ResNet/" + "hallucinated_" + str(e) + "/generated_"+ str(i) + ".png"
            plt.savefig(filename)

tb.close()
torch.save({'epoch': e, 'optimizer_state_dict_D': optimizerD.state_dict(),
                    "optimizer_state_dict_G": optimizerG.state_dict()}, os.path.join(folder_name,'checkpoint.pth'))
torch.save(g.state_dict(), os.path.join(folder_name, "gen_gr_ResN_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
torch.save(d.state_dict(), os.path.join(folder_name, "dis_gr_ResN_batch_" + batch_size_str + "_wd" + w_decay_str + "_lr" + lrate_str + "_e" + str(e) + ".pth"))
print("finished training")
