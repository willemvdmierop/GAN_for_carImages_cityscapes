import os, sys, re
import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.utils as utils
from torch.utils import data
import torchvision
from torchvision import transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
import skimage
from skimage.measure import find_contours
import random
import ssl
import ds_interface
import util
import list_of_cityscapes_labels as lab

#what is this?
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

#set device
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

#working directory
wd = '/Users/willemvandemierop/Documents/Master AI/Pycharm/DL Classification/Coursework'
images_dir = wd + "/leftImg8bit_trainvaltest/leftImg8bit/train/aachen"
gt_dir = wd + "/gtFine_trainvaltest/gtFine/train/aachen"

print("images dir",images_dir)
print("gt_dir", gt_dir)


img_max_size = [800, 800]
cityscapes_classes = lab.get_labels()
cityscapes_object_categories = []
cityscapes_to_new = {}
for value in cityscapes_classes:
    cityscapes_object_categories.append(value)

#for idz, c in enumerate(cityscapes_object_categories[1:]):
#    assert c in lab.label2id.keys()
#    cityscapes_to_new[lab.label2id[c]] = idz + 1


#print("Cityscapes Object Categories", cityscapes_object_categories, '\n',
# "Cityscapes Classes", cityscapes_classes)

# get MS COCO classes
coco_object_categories = []
coco_classes = {}
with open(os.path.join(wd, 'coco_labels.txt'), "r") as f:
    for id, l in enumerate(f.readlines()):
        # get the class label
        l = re.sub('[\d]+\W+', '', l.rstrip())
        # remove the old stuff
        if l == 'wood':
            break
        else:
            coco_object_categories.append(l)
            coco_classes[l] = id

#print("COCO Object Categories", coco_object_categories, '\n',
# "Coco Classes",coco_classes)


maskrcnn_args = {'num_classes': 91}
maskrcnn_model = model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, **maskrcnn_args)
#print(maskrcnn_model.forward)
weights = {}
for _n, par in maskrcnn_model.named_parameters():
    weights[_n] = par
#print(weights)
maskrcnn_model.eval()


################# testing the pretrained model on an image from cityscapes in aachen #################
def test_mask_on_City(im):
    im_path = im
    threshold = 0.75
    img = np.array(PILImage.open(im))
    plt.imshow(img)
    img = transforms.ToTensor()(img)
    #print("Image Size:", img.size())
    output = maskrcnn_model([img])
    #print('NN\'s Output', output)
    scores = output[0]['scores']
    bboxes = output[0]['boxes']
    classes = output[0]['labels']
    mask = output[0]['masks']
    #print("Mask Shape:", mask[0].shape)
    color_array = np.zeros([mask[0].shape[1], mask[0].shape[2], 3], dtype=np.uint8)
    #print("Color:", color_array, '\n', "color array shape:", color_array.shape)
    best_idx = np.where(scores > threshold)
    # we keep the best predictions
    best_scores = scores[best_idx]
    best_bboxes = bboxes[best_idx]
    best_classes = classes[best_idx]
    best_mask = mask[best_idx]
    util.plot_instance(image_path=im_path, best_bboxes=best_bboxes, best_classes=best_classes, best_mask=best_mask, \
                  color_array=color_array, best_scores=best_scores, class_names=coco_object_categories)

test_mask_on_City("/Users/willemvandemierop/Documents/Master AI/Pycharm/DL Classification/Coursework/v_03_with_carimages/cars/aachen_000020_000019_carImage.png")

#classes we want to predict in a picture 
list_of_predicted_classes = ['car']
# insert '__bgr__' class
list_of_predicted_classes.insert(0, '__bgr__')

city_args = {'classes': cityscapes_classes, 'dir': images_dir, 'GT_dir': gt_dir,'img_max_size': img_max_size, 'predicted_classes': list_of_predicted_classes}
cityscapes_data_point = ds_interface.CityScapes(**city_args)

# ds_interface.__getitem__: returns 'index', 'X' (input image), 'y' (tuple of lists
# containing objects' labels and their corresponding ground truth mask coordinates 
# from json file)

#print(cityscapes_data_point.__getitem__(0))

# TRAIN
dataloader_args = {'batch_size': 1, 'shuffle': True}
dataloader = data.DataLoader(cityscapes_data_point, **dataloader_args)
total_epoch = 1

maskrcnn_model.train()

if device == torch.device('cuda'):
   maskrcnn_model = maskrcnn_model.to(device)

#Is there a reason for SGD?
optimizer_pars = {'lr':1e-5, 'weight_decay':1e-3}
optimizer = torch.optim.SGD(list(maskrcnn_model.parameters()),**optimizer_pars)

# TO DO: arrange training loop according to cityscapes'
# dataset > see ds_interface.__getitem__()
for e in range(total_epoch):
    for id, batch in enumerate(dataloader):
        optimizer.zero_grad()
        idx, X, y = batch
        #print(idx, y)
        if device == torch.device('cuda'):
           X, y['labels'],y['boxes'],y['masks'] = X.to(device), y['labels'].to(device), y['boxes'].to(device), y['masks'].to(device)
        # list of images
        images = [im for im in X]
        #todo we have 45 labels and 31 masks?? also list of labels
        #todo maybe annToMask?
        boxes = y['boxes'] #todo this returns polygons instead of bboxes
        targets = []
        lab={}
        # get rid of the first dimension (batch)
        # if you have >1 images, make another loop
        lab[''] = y.keys().squeeze_(0)
        lab['la']=y['labels'].squeeze_(0)
        targets.append(lab)
        # avoid empty objects
        if len(targets)>0:
           loss = frcnn_model(images, targets) #todo maskrcnn_model?
           total_loss = 0
           for k in loss.keys():
               total_loss += loss[k]
           print(total_loss)
           total_loss.backward()
           optimizer.step()
