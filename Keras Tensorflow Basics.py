# In this section we will import the layers needed to execute the code

import tensorflow
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# First, lets grab some data. We need samples and their corresponding labels.

# Initialize the label and sample arrays

train_labels = []
train_samples = []

# initialize the dataset.
# Example Data: An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial.
# The trial had 2100 participants. Half were under 65 years old, half were 65 or older.
# 95% of patients 65 or older experienced side effects.

for i in range(50):
    # the 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # the 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # the 95% of younger people who did not experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # the 95% of older people who did experience side effects
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)

