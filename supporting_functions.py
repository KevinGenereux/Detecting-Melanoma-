""" Imports """
import numpy as np
import pandas as pd

import re

import keras
import tensorflow as tf


from keras.preprocessing import image
from keras.utils import to_categorical
from keras.applications.vgg16 import VGG16,preprocess_input
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten,BatchNormalization,Activation
from keras.preprocessing.image import ImageDataGenerator


from keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras import backend as K

from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

from kaggle_datasets import KaggleDatasets



""" Setup """

# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

# Configuration
BATCH_SIZE = 8 * 8 # kind of a hack. We should have access to 8 TPUs
IMAGE_SIZE = [768, 768]
NUM_CLASSES = 1

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')



""" Supporting Functions """

def read_tfrecord(data):
  
    features = {
    # tf.string = byte string (not text string)
    "image": tf.io.FixedLenFeature([], tf.string), # shape [] means scalar, here, a single byte string
    "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar, i.e. a single item
    }

    # decode the TFRecord
    tf_record = tf.io.parse_single_example(data, features)

    # Typical code for decoding compressed images
    image = tf.image.decode_jpeg(tf_record['image'], channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    
    target = tf_record['target']

    return image, target


def get_training_dataset():
    
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(TRAINING_FILENAMES, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    
    return dataset

def get_testing_dataset():
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(TEST_FILENAMES, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    dataset = dataset.repeat()
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    
    return dataset


def get_sizes():
    
    return IMAGE_SIZE, BATCH_SIZE



""" Model Functions """

def vgg16_model(num_classes=None):
    model = VGG16(weights='imagenet',include_top=False,input_shape=(224,224,3))
    x = Flatten()(model.output)
    output = Dense(1,activation='sigmoid')(x)
    model = Model(model.input,output)
    
    return model

"""def efiNet_model():
    model = efn.EfficientNetB5(input_shape=(*IMAGE_SIZE, 3), weights='imagenet', include_top=False)
    return model"""