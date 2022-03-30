#https://www.tensorflow.org/tutorials/keras/regression#normalization

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
#url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

#column_names = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides',
 #               'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']

#dataset = pd.read_csv(url, names=column_names, na_values='?', sep=';')

#LOAD AND SPLIT DATASET
dataset = pd.read_csv('winequality-red.csv', sep=';', )

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#USE A CORRELATION MATRIX AND VISUALISE IT USING HEATMAP TO SEE POSITIVE CORRELATION OF QUALITY WITH OTHER FEATURES
#sns.heatmap(dataset.corr(), annot=True)
#plt.show()

    #Quality has most positive correlation with, in descending order of, alcohol, sulphates, citric acid, fixed acidity

#SPLITTING DATASET INTO FEATURES AND LABELS
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('quality')
test_labels = test_features.pop('quality')

#NORMALIZATION LAYER TO ACCOMODATE DIFFERENT SCALES AND RANGES
normalizer = tf.keras.layers.Normalization(axis=-1)
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers/Normalization
    #adapt() before fit() or predict()
normalizer.adapt(np.array(train_features))

##LINEAR REGRESSION WITH ONE VARIABLE

    #First, create a NumPy array made of the 'alcohol' features. Then, instantiate the tf.keras.layers.Normalization
    #and fit its state to the alcohol data
    #this model will predict 'quality' from 'alcohol'

alcohol = np.array(train_features['alcohol'])

    # Training a model with tf.keras typically starts by defining the model architecture.
    # Use a tf.keras.Sequential model, which represents a sequence of steps.
    # There are two steps in your single-variable linear regression model:
    # Normalize the 'alcohol' input features using the tf.keras.layers.Normalization preprocessing layer.
    # Apply a linear transformation () to produce 1 output using a linear layer (tf.keras.layers.Dense).

alcohol_normalizer = layers.Normalization(input_shape=[1,], axis=None)
alcohol_normalizer.adapt(alcohol)

    #Build the Keras Sequential model:
alcohol_model = tf.keras.Sequential([
    alcohol_normalizer,
    layers.Dense(units=1)
])

#print(alcohol_model.predict(alcohol[:10]))

    #configure the training procedure using the Keras Model.compile method.
    #The most important arguments to compile are the loss and the optimizer, since these define what will be optimized
    #(mean_absolute_error) and how (using the tf.keras.optimizers.Adam).

alcohol_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error')

    #Use Keras Model.fit to execute the training for 100 epochs:
history = alcohol_model.fit(
    train_features['alcohol'],
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
#print(hist.tail())

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [alcohol]')
    plt.legend()
    plt.grid(True)
    plt.show()

#plot_loss(history)

test_results={}

test_results['alcohol_model'] = alcohol_model.evaluate(
    test_features['alcohol'],
    test_labels, verbose=0)

x = tf.linspace(0.0, 250, 251)
y = alcohol_model.predict(x)

def plot_alcohol(x, y):
    plt.scatter(train_features['alcohol'], train_labels, label='Data')
    plt.plot(x, y, color='k', label='Predictions')
    plt.xlabel('Alcohol')
    plt.ylabel('Quality')
    plt.legend()
    plt.show()


#plot_alcohol(x,y)

linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
])

#print(linear_model.predict(train_features[:10]))

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss='mean_absolute_error'
)

history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=0,
    validation_split=0.2)

#plot_loss(history)

test_results['linear_model'] = linear_model.evaluate(
    test_features, test_labels, verbose=0)


def build_and_compile_model(norm):
    model = keras.Sequential([
        norm,
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        loss='mean_absolute_error')
    return model


dnn_alcohol_model = build_and_compile_model(alcohol_normalizer)
#dnn_alcohol_model.summary()

history=dnn_alcohol_model.fit(
    train_features['alcohol'],
    train_labels,
    validation_split=0.2,
    verbose=0,
    epochs=100
)

x = tf.linspace(0.0, 250, 251)
y = dnn_alcohol_model.predict(x)

#plot_alcohol(x, y)

test_results['dnn_alcohol_model']=dnn_alcohol_model.evaluate(
    test_features['alcohol'], test_labels, verbose=0
)

dnn_model = build_and_compile_model(normalizer)
dnn_model.summary()
history = dnn_model.fit(
    train_features,
    train_labels,
    validation_split=0.2,
    verbose=0, epochs=100)

test_results['dnn_model'] = dnn_model.evaluate(test_features, test_labels, verbose=0)

pd.DataFrame(test_results, index=['Mean absolute error']).T

test_predictions = dnn_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [quality]')
plt.ylabel('Predictions [quality]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)

error = test_predictions - test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error [quality]')
_ = plt.ylabel('Count')

plt.show()
#dnn_model.save('dnn_model')