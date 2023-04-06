########################################################################
# This file will show the basics of how to perform data augmentation using keras tensorflow
########################################################################

import matplotlib.pyplot as plt
import numpy as np
import os
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

########################################################################
# Create the needed methods (this one is directly from tensorflow)
########################################################################
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

########################################################################
# Time to augment some data
########################################################################

gen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.15, zoom_range=0.1,
    channel_shift_range=10., horizontal_flip=True)

# All the parameters being passed are the different ways we're telling Keras to augment the image.
# See this link for the documentation on this method and the augmentation units required:
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator

# Now we will grab a random image of a dog from disk

chosen_image = random.choice(os.listdir(r'C:\Users\brmar\PycharmProjects\Deep-Learning-\Data\Dogs and Cats\train\train\dog'))

# We then create a variable called image_path and set that to the relative location on disk of the chosen image.

image_path = r'C:\Users\brmar\PycharmProjects\Deep-Learning-\Data\Dogs and Cats\train\train\dog/' + chosen_image

'''
Next, we'll obtain the image by reading the image from disk by using 
plt.imread() and passing in the image_path. We also, expand the dimensions so that the 
image is compatible for how we'll use it later.' 
'''

image = np.expand_dims(plt.imread(image_path),0)

# now we plot to see what it looks like

plt.imshow(image[0])

# Now we will generate blocks of augmented images
# Step 1, create an iterator

aug_iter = gen.flow(image)

# The flow() function takes numpy data and generates batches of augmented data.
# Now we'll get ten samples of the augmented images.
aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]

plotImages(aug_images)

'''
Note, if you'd like to save these images so that you can add them to your training set, then to gen.flow(), 
you should also specify the parameter save_to_dir and set it equal to a valid location on disk.
You can optionally specify a prefix for which to prepend to file names of the saved augmented images, 
as well as optionally specify the file type as 'png' or 'jpeg' images. 'png' is the default.
'''

# Create the new file path for the augmented data directory

#os.chdir('Data/Dogs and Cats/')
#if os.path.isdir('Dogs and Cats/augmented') is False:
#    os.makedirs('augmented')
#os.chdir('../../')

aug_iter = gen.flow(image, save_to_dir=r'C:\Users\brmar\PycharmProjects\Deep-Learning-\Data\Dogs and Cats\augmented'
                    , save_prefix='aug-image-', save_format='jpeg')