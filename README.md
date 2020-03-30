# Different GAN structures for generating Car Images from cityscapes dataset
This repository generates car images from the cityscapes dataset with different GAN structures. The car images are generated using the gt mask labels from the cityscapes dataset, in particular the fine annotation dataset. Then a mask R-CNN is used to see if the mask R-CNN is able to predict a car, when it is able to predict a car then the image will be saved. The mask R-CNN will make sure that some car images that are obstructed by object will not be saved. This is done with the generate_car_images_cityscapes.py file. 

These images are the dataset for the different GAN structues that are build. 
