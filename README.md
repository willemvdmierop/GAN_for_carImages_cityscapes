# Different GAN structures for generating Car Images from cityscapes dataset
This repository generates car images from the cityscapes dataset with different GAN structures. The car images are generated using the gt mask labels from the cityscapes dataset, in particular the fine annotation dataset. Then a mask R-CNN is used to see if the mask R-CNN is able to predict a car, when it is able to predict a car then the image will be saved. The mask R-CNN will make sure that some car images that are obstructed by object will not be saved. This is done with the generate_car_images_cityscapes.py file. 

The original image and the final image after preprocessing:

<img src="images/aachen_000000_000019_leftImg8bit.png" width="420"/> <img src="images/aachen_000000_000019_carImage_zoomed.png" width="420"/> 

Below is a picture shown where the car is obstructed and that because of the Mask R-CNN does not get saved. 

<img src="images/aachen_000000_000019_leftImg8bit.png" width="420">
These images are the dataset for the different GAN structues that are build. 
