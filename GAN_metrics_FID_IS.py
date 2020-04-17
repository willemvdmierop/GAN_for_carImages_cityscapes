import sys
import os
import time
import torch
from imageio import imread
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
import pandas as pd
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x):
        return x

from inception import InceptionV3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# This code is based on: https://github.com/mseitzer/pytorch-fid

def get_activations(files, model, batch_size=50, dims=2048, cuda=False, verbose=False):
    model.eval()
    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)
    pred_arr = np.empty((len(files), dims))

    for i in tqdm(range(0, len(files), batch_size)):
        start = i
        end = i + batch_size
        images = np.array([imread(str(f))[:, :, :3].astype(np.float32) for f in files[start:end]])
        # reshape images to (n_images, 3, height, width)
        images = images.transpose((0, 3, 1, 2))
        images /= 255
        batch = torch.from_numpy(images).type(torch.cuda.FloatTensor)
        if cuda:
            batch = batch.to(device)
        pred = model(batch)[0]
        # If model output is not scaler, apply global spatial average pooling.
        # This happens if dimensionality not equal to 2048
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))
        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)
    return pred_arr


def calculate_activation_statistics(files, model, batch_size=50, dims=2048, cuda=False, verbose=False):
    act = get_activations(files, model, batch_size, dims, cuda, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def compute_statistics_of_path(path, model, batch_size, dims, cuda):
    files = []
    single = False
    if not os.path.isdir(path):
        raise NameError
    else:
        for filename in sorted(os.listdir(path)):
            files.append(os.path.join(path, filename))
    mu1, sigma1 = calculate_activation_statistics(files, model, batch_size, dims, cuda)
    return mu1, sigma1


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)  # Convert inputs to arrays with at least one dimension
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
        # .imag = """The imaginary part of the array.
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


def calculate_FID_given_paths(path1, path2, batch_size, cuda, dims):
    if not os.path.exists(path1):
        raise RuntimeError("Invalid path: {path1}")
    if not os.path.exists(path2):
        raise RuntimeError("Invalid path: {path2}")

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])
    if cuda:
        model.cuda()
    mu1, sigma1 = compute_statistics_of_path(path1, model, batch_size, dims, cuda)
    mu2, sigma2 = compute_statistics_of_path(path2, model, batch_size, dims, cuda)

    FID = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return FID


def inception_score(path_imgs, inception_model, cuda=True, batch_size=32, resize=False, splits=1):
    files = []
    for filename in sorted(os.listdir(path_imgs)):
        files.append(os.path.join(path_imgs, filename))
    # print(files)
    for i in range(0, len(files)):
        images = np.array([imread(str(f))[:, :, :3].astype(np.float32) for f in files])
    images = images.transpose((0, 3, 1, 2))
    images /= 255
    imgs = images
    # print(imgs.shape)
    N = len(imgs)
    # print(N)
    # print(batch_size)
    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_model
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = x.to(device)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


# path_fake = "/content/drive/My Drive/DL Classification (705)/GANs/Fake_images/Fake_4500"
# IS_mean, IS_std = inception_score(path_imgs=path_fake,cuda = True, batch_size = 5)
# print("IS mean",IS_mean, "IS_std", IS_std)
def calculate_FID_compared_to_real(path, mu_real, sigma_real, model, batch_size, cuda, dims):
    if not os.path.exists(path):
        raise RuntimeError("this is a wrong path")

    model = model
    if cuda:
        model.cuda()
    mu1, sigma1 = mu_real, sigma_real
    mu2, sigma2 = compute_statistics_of_path(path, model, batch_size, dims, cuda)

    FID = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

    return FID


def statistics_real_images(path, batch_size, model, cuda):
    model = model
    mu1, sigma1 = compute_statistics_of_path(path, model, batch_size, dims, cuda)
    return mu1, sigma1

wd = os.getcwd()
path_real = os.path.join(wd,"Real_images_Green")
path_fake = os.path.join(wd,"gen_images_green_DC")

######################################## parameters ############################
batch_size = 1
cuda = True
dims = 2048
saving_name_metrics = "Metrics_GAN.csv"
epoch_start, epoch_end = 500, 4000
########################################## models ##############################
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
model_FID = InceptionV3([block_idx])
model_FID = model_FID.to(device)

inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
inception_model.eval()
inception_model = inception_model.to(device)

################################## statistics real images ######################
mu_real, sigma_real = statistics_real_images(path_real, batch_size, model=model_FID, cuda=cuda)

################################## computation of metrics ######################
for filename in sorted(os.listdir(path_real)):
    if filename == '.DS_Store':
        os.remove(path_real + "/" + filename)

FID_list = []
IS_list = []
IS_STD_list = []
for f in sorted(os.listdir(path_fake)):
    start = time.time()
    if f == '.DS_Store':
        os.remove(path_fake + "/" + f)
        continue
    path_generated_images = os.path.join(path_fake, f)
    print(path_generated_images)
    FID = calculate_FID_compared_to_real(path=path_generated_images, mu_real=mu_real, sigma_real=sigma_real,
                                         model=model_FID, batch_size=1, cuda=cuda, dims=dims)
    print("The FID for our generated images is {}, calculation of FID took {} seconds".format(FID, time.time() - start))
    IS_mean, IS_std = inception_score(path_imgs=path_generated_images, inception_model=inception_model, cuda=cuda,
                                      batch_size=batch_size)
    print("The inception score mean score is {}, IS std is {} ,calculation of IC took {} s".format(IS_mean, IS_std,
                                                                                                   time.time() - start))
    FID_list.append(FID)
    IS_list.append(IS_mean)
    IS_STD_list.append(IS_std)

df = pd.DataFrame(FID_list, columns=['FID'])
df['Inception Score'] = IS_list
df['IS std'] = IS_STD_list
df.to_csv(saving_name_metrics)


epochs = np.arange(epoch_start, epoch_end, 100)
x = epochs[:(len(FID_list))]
fig, ax = plt.subplots(2,1,figsize = (20,20))

ax[0].plot(x,df['FID'])
ax[0].set_title('FID Score of generated images')
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("FID")

ax[1].plot(x,df['Inception Score'])
ax[1].set_title('Inception Score of generated images')
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("IS")

