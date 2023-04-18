
'''
This file will apply the concepts of Fine-Tuning to the MobileNet model
using a custom dataset obtained here (greyscale): https://www.kaggle.com/datasets/ardamavi/sign-language-digits-dataset
and here (RGB): https://github.com/ardamavi/Sign-Language-Digits-Dataset

Here is a link to the MobileNets paper: https://arxiv.org/pdf/1704.04861.pdf

'''
########################################################################
# Imports
########################################################################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
from IPython.display import Image
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt

########################################################################
# Load the MobileNet model from Internet or Disk
########################################################################

mobile = tf.keras.applications.mobilenet.MobileNet()
print(mobile.summary())

# This will save the model to disk
mobile.save('models/MobileNet.h5')

# Use the load model method when you want to use this model from disk instead of downloading it again
#mobile = load_model('MobileNet.h5')
#print(mobile.summary())

########################################################################
# Organize the RGB Data
########################################################################

# Once this code has been run on the downloaded dataset, it can be commented out

# Organize data into train, valid, test dirs

'''
First, we create train, valid, and test directories.
We then move all class directories (0 - 9) with their respective images from Sign_Language_Digits_Dataset
into the train directory.
We then make class directories (0 - 9) for the valid and test data sets as well.
We then loop through each class directory in the train directory and randomly move 30 images from each class into 
the corresponding class directory in valid and 5 images from each class into the corresponding class directory in test.

Finally, we end by moving back into the current directory (Not necessary if not using a Jupyter Notebook).
'''
os.chdir('Data/Sign_Language_Digits_Dataset_RGB')
if os.path.isdir('train/0/') is False:
    os.mkdir('train')
    os.mkdir('valid')
    os.mkdir('test')

    for i in range(0, 10):
        shutil.move(f'{i}', 'train')
        os.mkdir(f'valid/{i}')
        os.mkdir(f'test/{i}')

        valid_samples = random.sample(os.listdir(f'train/{i}'), 30)
        for j in valid_samples:
            shutil.move(f'train/{i}/{j}', f'valid/{i}')

        test_samples = random.sample(os.listdir(f'train/{i}'), 5)
        for k in test_samples:
            shutil.move(f'train/{i}/{k}', f'test/{i}')
# os.chdir('../..')

'''
After our image data is all organized on disk, we need to create the directory iterators for the train, validation,
and test sets in the exact same way as we did for the cat and dog data set in the CNN Basics and Fine-Tuning Basics
Files.

'''

train_path = 'Data/Sign_Language_Digits_Dataset_RGB/train'
valid_path = 'Data/Sign_Language_Digits_Dataset_RGB/valid'
test_path = 'Data/Sign_Language_Digits_Dataset_RGB/test'

train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=valid_path, target_size=(224,224), batch_size=10)
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input).flow_from_directory(
    directory=test_path, target_size=(224,224), batch_size=10, shuffle=False)

'''
Notice the preprocessing_function parameter we're supplying to ImageDataGenerator. 
We're setting this equal to tf.keras.applications.mobilenet.preprocess_input. 
This is going to do the necessary MobileNet preprocessing on the images obtained from flow_from_directory().

To review, preprocessing images for MobileNet requires scaling the pixel values in the image between -1 and 1, and 
returning the preprocessed image data as a numpy array.

To flow_from directory(), we're passing in the path to the data set, the target_size to resize the images to, 
and the batch_size we're choosing to use for training. We do this exact same thing for all three data sets.

For test_batches, we're also supplying one additional parameter, shuffle=False, 
which causes the test dataset to not be shuffled, so that we can access the corresponding non-shuffled test labels 
to plot to a confusion matrix later.

'''