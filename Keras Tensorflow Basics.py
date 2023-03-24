########################################################################
# In this section we will import the libraries needed to execute the code
########################################################################
# These are the libraries needed to create and process the training dataset and labels
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt

# These are the libraries needed to create the deep neural network model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.metrics import categorical_crossentropy


########################################################################
# In this section we will create a dataset along with corresponding labels for training
########################################################################

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
# finally, we will scale the training samples down to between 0 and 1

train_samples = np.array(train_samples)
train_labels = np.array(train_labels)
train_samples, train_labels = shuffle(train_samples, train_labels)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))

# let's print out the scaled train samples
#for i in scaled_train_samples:
    #print(i)


# time to create the test set. This is the data that the trained model will perform inference on

test_labels =  []
test_samples = []

for i in range(10):
    # The 5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # The 5% of older individuals who did not experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # The 95% of younger individuals who did not experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # The 95% of older individuals who did experience side effects
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

########################################################################
# Now, we will create the sequential model using Tensorflow
########################################################################

# First, let's check to see if Tensorflow recognizes that we have a GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))


# Now, let's build the model.  Using the Sequential object, tf.keras will build a sequential stack of layers.
# Sequential Object accepts a list, and each element of the list should be a layer

# notice that the input layer is not strictly specified beyond the input_shape argument
# in this example, only the 3 dense layers are defined within the list fed to the Sequential Object

model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

# Let's print the model out to see what it looks like

print(model.summary())

########################################################################
# Now, we will train the Neural Network
########################################################################

# Now that the model has been created, we must compile it and specify the optimizer,
# learning rates, and metrics

model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Time to train using the fit function. The first parameter, X, is our input data. Y is
# our label data. The batch size is specified next, followed by the number of epochs,
# and the shuffle parameter. Finally, verbose is set to 2 so that we can see the results of each epoch.

# model.fit(x=scaled_train_samples, y=train_labels, batch_size=10, epochs=30, verbose=2)

# Let's now make a validation dataset that will allow us to test our model
# With data that it has not been trained on
# There are two ways to do this using Model.Fit
# One way is to assign the validation data argument to a pre-made validation dataset
# The other is to assign the validation split argument to a decimal between 0 and 1
# This will split the training set into a fraction to be used as the validation set
# Since the validation data is selected before the default shuffling in model.fit,
# We need to make sure that the dataset is shuffled ahead of time (which we did when
# We created the dataset)

model.fit(
      x=scaled_train_samples,
      y=train_labels,
      validation_split=0.1,
      batch_size=10,
      epochs=30,
      shuffle=True,
      verbose=2)

# With this training set up, we will see loss and accuracy metrics
# for the training and validation datasets

# Now it is time to make predicitons with the trained model on the test set
# To do this, we use the predict method, passing the test set as input
# The output of this method is an array of predictions

predictions = model.predict(
      x=scaled_test_samples
    , batch_size=10
    , verbose=0
)

# let's take a look at the predicitons array

for i in predictions:
    print(i)

# this array consists of predicted probabilities of both classes for each inference performed by the model
# on the test data. notice that the probabilities add up to 1.
# this next piece shows what the model predicitons are. I.e. the model chooses the max probability as its answer.
# Notice that even if the predicted probabilities are close, the model will choose the higher probability.

rounded_predictions = np.argmax(predictions, axis=-1)

#for i in rounded_predictions:
    #print(i)

###################################################################
# In this section, we will create a confusion matrix to determine how accurately the model performed
# on the test dataset
###################################################################

# First, define the confusion matrix method. This method is from SKLearn's website

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Next, we will create the confusion matrix itself, using the SKLearn method

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

# Next, we will define the labels for the confusion matrix plot

cm_plot_labels = ['no_side_effects','had_side_effects']

# Finally, we will invoke the method and array above and create the confusion matrix

plot_confusion_matrix(cm=cm, classes=cm_plot_labels, title='Confusion Matrix')

###################################################################
# In this section, we will save and load the model as an example for future work
###################################################################

# this will save a model at its current state after it was trained so that we could make use of it later.
# we pass the file name that we want to save it as

model.save('models/medical_trial_model.h5')

# This method of saving will save everything about the model –
# the architecture, the weights, the optimizer, the state of the optimizer, the learning rate, the loss, etc.
# We can then load this model and verify that it has the same attributes as the previous model
# We can also inspect attributes about the model, like the optimizer and loss by calling
# model.optimizer and model.loss on the loaded model and compare the results to the previously saved model.

new_model = load_model('models/medical_trial_model.h5')
print(new_model.summary())
print(new_model.optimizer())
print(new_model.loss())

# There is another way we save only the architecture of the model.
# This will not save the model weights, configurations, optimizer, loss or anything else.
# This only saves the architecture of the model.
# We can do this by calling model.to_json().
# This will save the architecture of the model as a JSON string.
# If we print out the string, we can see exactly what this looks like.

json_string = model.to_json()
print(json_string)

# Now that we have this saved, we can create a new model from it.
# First we'll import the needed model_from_json function, and then we can load the model architecture.

model_architecture = model_from_json(json_string)
print(model_architecture.summary())

# We can also only save the weights of the model.
# We can do this by calling model.save_weights() and passing in the path and file name
# to save the weights to with an h5 extension.

model.save_weights('models/my_model_weights.h5')

#At a later point, we could then load the saved weights in to a new model,
# but the new model will need to have the same architecture as the old model before the weights can be saved.

model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model2.load_weights('models/my_model_weights.h5')

# now we can use this model just like the first model we trained, as it has the exact same weights as the trained
# model.

