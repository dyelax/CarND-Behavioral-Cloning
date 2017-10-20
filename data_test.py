import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import os
from scipy.misc import imsave

from utils import gen_batches

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('imgs_dir', 'data/sdc-lab//IMG/', 'The directory of the image data.')
flags.DEFINE_string('data_path', 'data/mjc+mrg/combined_hal.csv', 'The path to the csv of training data.')
flags.DEFINE_string('save_dir', 'save/hal4/', 'The directory to which to save the model.')
flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
flags.DEFINE_integer('num_epochs', 10, 'The number of epochs to train for.')
flags.DEFINE_float('lrate', 0.00001, 'The learning rate for training.')


def main(_):
    ##
    # Load Data
    ##

    with open(FLAGS.data_path, 'r') as f:
        reader = csv.reader(f)
        # data is a list of tuples (img path, steering angle)
        data = np.array([row for row in reader])

    # Split train and validation data
    np.random.shuffle(data)
    split_i = int(len(data) * 0.9)
    X_train, y_train = list(zip(*data[:split_i]))
    X_val, y_val = list(zip(*data[split_i:]))

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    gen = gen_batches(X_train, y_train, 10)
    imgs, labels = gen.__next__()

    print(imgs[0])

    for i, img in enumerate(imgs):
        imsave('test/' + str(i) + '.png', np.squeeze(img))

if __name__ == '__main__':
    tf.app.run()
