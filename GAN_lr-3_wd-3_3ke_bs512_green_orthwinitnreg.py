import os
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
from torch import optim
from torch.nn import BCELoss
from torch.utils import data
from torchvision import transforms
#from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

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
			transforms.Normalize(mean = [0.485, 0.457, 0.407],
								 std = [1, 1, 1])
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
		self.init_weights()
		self.main = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(in_channels=self.nz, out_channels=self.ngf * 8, kernel_size=4, stride=1, padding=0,
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

	# Orthogonal weight initialization (source: https://arxiv.org/abs/1312.6120)
	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.ConvTranspose2d):
				init.orthogonal_(module.weight)

	def forward(self, input):
		return self.main(input)


# Discriminator takes an 'image': object dimensionality batch_size x 3 x H x W
class Discriminator(nn.Module):
	def __init__(self, **kwargs):
		super(Discriminator, self).__init__()
		self.ndf = kwargs['ndf']
		self.nc = kwargs['nc']
		self.init_weights()
		self.main = nn.Sequential(
			# input is (nc) x 64 x 64
			nn.Conv2d(in_channels=self.nc, out_channels=self.ndf, kernel_size=4, stride=2, padding=1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 32 x 32
			nn.Conv2d(self.ndf, self.ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 16 x 16
			nn.Conv2d(self.ndf * 2, self.ndf * 4, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 8 x 8
			nn.Conv2d(self.ndf * 4, self.ndf * 8, 4, 2, 1, bias=False),
			nn.BatchNorm2d(self.ndf * 8),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*8) x 4 x 4
			nn.Conv2d(self.ndf * 8, 1, 4, 1, 0, bias=False),
			nn.Sigmoid()
		)
		#self.classifier = nn.Sequential(
		#	nn.Linear(512 * 11 * 11, 1),
		#	nn.Sigmoid()
		#)

	# Orthogonal weight initialization (source: https://arxiv.org/abs/1312.6120)
	def init_weights(self):
		for module in self.modules():
			if isinstance(module, nn.ConvTranspose2d):
				init.orthogonal_(module.weight)

	def forward(self, input):
		features = self.main(input)
		# features = features.view(-1, features.size()[1]*features.size()[2]*features.size()[3])
		# out=self.classifier(features)
		return features

# orthogonal parameter regularization. Source: https://github.com/ajbrock/BigGAN-PyTorch/blob/master/utils.py
def ortho_reg(model, strength = 1e-4, blacklist = []):
	with torch.no_grad():
		for param in model.parameters():
			# Regularize only parameters with at least 2 dim or whitelisted
			if len(param.shape) < 2 or any([param is item for item in blacklist]):
				continue
			# take all parameters to matrix shape, i.e. 2 dim
			w = param.view(param.shape[0], -1)
			# apply orthogonal regularization
			grad = (2 * torch.mm(torch.mm(w, w.t())
								 * (1. - torch.eye(w.shape[0], device = w.device)), w))
			# update regularized parameters
			param.grad.data += strength * grad.view(param.shape)


tb = SummaryWriter()
batch_size = 512
image_size = [64, 64]
assert len(image_size) == 2

wd = os.getcwd()
#if not os.path.exists(os.path.join(wd, 'checkpoints')):
#	os.mkdir(os.path.join(wd, 'checkpoints'))
#checkpoints = os.path.join(wd, 'checkpoints')

# number of generator feature filters
ngf = 64
# number of discriminator feater filters
ndf = 64
# number of channels
nc = 3  # ('RGB')
# latent space (z) size: G input
nz = 100

num_epochs = 3000

# dataset and dataloader
dataset_pars = {'img_size': image_size,
				'img_data': '/Users/willemvandemierop/Google Drive/DL Classification (705)/v_03_with_carimages/cars3_green'}
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
g = Generator(**g_pars)
d = Discriminator(**d_pars)

# load pretrained model?
load_pretrained = None
if load_pretrained:
	pretrained_g = os.path.join(checkpoints, 'G_lr-3_wd-3_3ke_transp_orthwinit.pth')
	pretrained_d = os.path.join(checkpoints, 'D_lr-3_wd-3_3ke_transp_orthwinit.pth')
	g.load_state_dict(torch.load(pretrained_g))
	d.load_state_dict(torch.load(pretrained_d))

g = g.to(device)
d = d.to(device)
# print(g)
# print(d)

# define discriminator and generator update steps
d_steps = 2
g_steps = 1

# create labels
real_label = 1
generated_label = 0

# optimizers
lrate = 1e-3
optimizer_pars = {'lr': lrate, 'weight_decay': 1e-3}
optimizerD = optim.Adam(d.parameters(), **optimizer_pars)
optimizerG = optim.Adam(g.parameters(), **optimizer_pars)

# loss function
# loss = BCEWithLogitsLoss()
loss = BCELoss()
loss = loss.to(device)

# main train loop
for e in range(num_epochs):
	for id, imgs in dataloader:
		# print(id)
		# first, train the discriminator
		d.zero_grad()
		g.zero_grad()
		imgs = imgs.to(device)
		batch_size = imgs.size()[0]
		# labels: all 1s
		labels_real = torch.ones(batch_size).unsqueeze_(0)
		labels_real = labels_real.to(device)
		# get the prediction of the D model
		# D(x)
		d_real = d(imgs).view(1, -1)
		# loss from real data
		d_loss_real = loss(d_real, labels_real)
		#d_loss_real.backward()

		# generator input and output
		z = torch.randn(batch_size, nz, 1, 1, device = device)
		h = g(z)
		# all labels are 0s
		labels_fake = torch.zeros(batch_size).unsqueeze_(0)
		labels_fake = labels_fake.to(device)
		# D(1-G(z))
		d_fake = d(h.detach()).view(1, -1)
		# loss from generated data
		d_loss_fake = loss(d_fake, labels_fake)
		#d_loss_fake.backward()
		total_loss = d_loss_real + d_loss_fake
		total_loss.backward()
		# apply orthogonal regularization to discriminator parameters
		ortho_reg(d)
		# update discriminator parameters
		optimizerD.step()

		# D(G(z))
		g.zero_grad()
		d_fake = d(h).view(1, -1)
		g_loss = loss(d_fake, labels_real)
		# update generator weights
		g_loss.backward()
		# apply orthogonal regularization to generator parameters
		ortho_reg(g)
		# update generator parameters
		optimizerG.step()

		# track losses on tensorboard
		tb.add_scalar('Discriminator Loss (D(x) + D(G(z)))', total_loss, e)
		tb.add_scalar('Generator Loss (G(z))', g_loss, e)

		# save parameters checkpoint
	if e > 1500:
		if e%250 == 0:
			torch.save(g.state_dict(), os.path.join(checkpoints, 'G_lr-3_wd-3_{}e_green_orthwinitnreg.pth' .format(e)))
			torch.save(d.state_dict(), os.path.join(checkpoints, 'D_lr-3_wd-3_{}e_green_orthwinitnreg.pth' .format(e)))

tb.close()
torch.save(g.state_dict(), os.path.join(checkpoints, 'G_lr-3_wd-3_3ke_green_orthwinitnreg.pth'))
torch.save(d.state_dict(), os.path.join(checkpoints, 'D_lr-3_wd-3_3ke_green_orthwinitnreg.pth'))