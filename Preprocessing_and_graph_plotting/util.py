import os, re
import cv2
import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image as PILImage
import skimage
import colorsys
import list_of_cityscapes_labels as lab

# what classes do we want to predict?
list_of_predicted_classes = ['car', 'person', 'sky']
# insert '__bgr__' class
list_of_predicted_classes.insert(0, '__bgr__')
cityscapes_to_new = {}
new_to_cityscapes = {}

# need to relabel the classes, e.g. car from class 13 becomes 1
for idz, c in enumerate(list_of_predicted_classes[1:]):
    assert c in lab.label2id.keys()
    cityscapes_to_new[lab.label2id[c]] = idz + 1
    new_to_cityscapes[idz + 1] = lab.label2id[c]


# print(cityscapes_to_new)
# print(new_to_cityscapes)

# this is for the object detection
# fname must point to a .json file, e.g.
# aachen_000097_000019_gtFine_polygons.json
# Cityscapes stores polygons instead of bboxes,
# therefore I extracted a bbox from polygon vertices  

def extract_bboxes_cityscapes(fname, list_of_correct_labels):
    with open(fname) as f:
        cs = json.load(f)
    objects = cs['objects']
    classes = []
    bboxes = []
    # extract 
    for o in objects:
        classlabel = o['label']
        # avoid some labels
        if classlabel in list_of_predicted_classes:
            # extract the label and convert it to integer
            classes.append([lab.label2id[o['label']]])
            # extract the bounding box from the polygon
            x, y = zip(*o['polygon'])
            min_x, max_x = min(x), max(x)
            min_y, max_y = min(y), max(y)
            bbox = [min_x, min_y, max_x, max_y]
            bboxes.append(bbox)
    # return a label: class of the object and bbox gt
    label = {}
    classes = torch.tensor(classes, dtype=torch.int64)
    label['classes'] = classes
    label['bboxes'] = torch.tensor(bboxes, dtype=torch.float32)
    return label


# print(extract_bboxes_cityscapes(os.path.join('/Users/ckxz/Desktop/@City/705, CV/CW/gt' ,'aachen_000027_000019_gtFine_polygons.json'),
# lab.label2id.keys()))

# this must point to a mask image, e.g. aachen_000096_000019_gtFine_labelIds.png
# this will return a full image mask with all non-selected classes=0 
# if k classes are selected for segmentation problem, there will be in total 
# k+1 labels  
def extract_segmentation_mask_cityscapes(fname, list_of_predicted_labels):
    mask = np.array(PILImage.open(fname))
    # get rid of classes we don't need, convert them to background
    for l in np.unique(mask):
        _l = lab.id2label[l]
        if not _l in list_of_predicted_labels:
            mask[mask == l] = 0

            # relabel the classes so that they are consecutive,
    # e.g. 0 for bgr, 1 for car, 2 for pedestrian, etc
    _cl = np.unique(mask)
    for c in _cl[1:]:
        mask[mask == c] = cityscapes_to_new[c]

    mask = torch.tensor(mask, dtype=torch.uint8)
    return mask


# print(extract_segmentation_mask_cityscapes(os.path.join('/Users/ckxz/Desktop/@City/705, CV/CW/gt' ,'aachen_000027_000019_gtFine_labelIds.png'),
# list_of_predicted_classes) )


# this must point to an instance mask, e.g. aachen_000096_000019_gtFine_instanceIds.png
# this will return a full image mask with all non-selected classes=0
# for mask prediction you should add the code here that identifies classes of objects
# you can add code here to extract bboxes
def extract_instance_mask_cityscapes(fname, list_of_predicted_labels):
    mask = torch.tensor(np.array(PILImage.open(fname)), dtype=torch.uint8)

    # get segmentation mask to find which classes to get rid of
    # plt.imshow(mask)
    # plt.show()
    # print(np.unique(mask))
    _fname_segment = re.sub('instanceIds', 'labelIds', fname)
    _mask_segmentation = extract_segmentation_mask_cityscapes(_fname_segment, list_of_predicted_labels)
    # get rid of all instances except the predicted
    mask_instances_bgr = mask * _mask_segmentation
    # plt.imshow(mask_instances_bgr)
    # plt.show()
    _instances = np.unique(mask_instances_bgr)
    list_of_masks = []
    for _m in _instances[1:]:
        _mask = torch.zeros(mask.shape, dtype=torch.uint8)
        _mask[mask_instances_bgr == _m] = 1
        # plt.imshow(_mask)
        # plt.show()
        list_of_masks.append(_mask)

    return list_of_masks


# print(extract_instance_mask_cityscapes(os.path.join('/Users/ckxz/Desktop/@City/705, CV/CW/gt', 'aachen_000027_000019_gtFine_instanceIds.png'),
# list_of_predicted_classes))


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def plot_instance(image_path, best_bboxes, best_classes, best_mask, color_array, best_scores, class_names):
    im_path = image_path
    N = best_bboxes.shape[0]
    # for the objects with scores>threshold:
    # add bbox and mask
    colors = np.array(random_colors(N))
    colors = colors * 225
    if len(best_bboxes[0]) > 0:
        bgr_img = cv2.imread(im_path)
        #plt.imshow(im)
        ax = plt.gca()
        # plot masks
        # Do not use pytorch tensors
        # before plotting, convert to numpy
        for id, b in enumerate(best_bboxes):
            found = best_mask[id][0].detach().clone().cpu().numpy()
            color_array[found > 0.5] = colors[id]
            # elif classes[id] == second_best:
            #   found = mask[id][0].detach().clone().cpu().numpy()
            #   color_array[found>0.5] = colors[id]
            rect = Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], linewidth=1.5, edgecolor='r', facecolor='none',
                             linestyle="dashed")
            # label
            class_id = best_classes[id]
            score = best_scores[id] if best_scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
            ax.text(b[0], b[1] + 8, caption, color='w', size=11, backgroundcolor='none')
            ax.add_patch(rect)

        added_image = cv2.addWeighted(bgr_img, 0.5, color_array, 0.5, 0)
        plt.imshow(added_image)
        plt.show()
