# In this section we will import the layers needed to execute the code

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

# convert the train labels and samples lists into numpy arrays
# then shuffle them to eliminate ordering
# finally, we will scale the training samples down to between 0 to 1
train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_samples, train_labels = shuffle(train_samples, train_labels)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# let's print out the scaled train samples
for i in scaled_train_samples:
    print(i)

