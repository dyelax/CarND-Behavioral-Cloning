import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten

from utils import gen_batches

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('imgs_dir', 'data/IMG/', 'The directory of the image data.')
flags.DEFINE_string('data_path', 'data/driving_log_clean.csv', 'The path to the csv of training data.')
flags.DEFINE_integer('batch_size', 128, 'The minibatch size.')
flags.DEFINE_integer('num_epochs', 5, 'The number of epochs to train for.')
flags.DEFINE_float('lrate', 0.001, 'The learning rate for training.')

def main(_):
    ##
    # Load Data
    ##

    with open(FLAGS.data_path, 'r') as f:
        reader = csv.reader(f)
        # data is a list of tuples (img path, steering angle)
        data = np.array([row for row in reader])

    # Split train and validation data

    split_i = int(len(data) * 0.9)
    X_train, y_train = list(zip(*data[:split_i]))
    X_val, y_val = list(zip(*data[split_i:]))

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_val, y_val = np.array(X_val), np.array(y_val)

    ##
    # Define Model
    ##

    model = Sequential([
        Conv2D(32, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
        Conv2D(64, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
        MaxPooling2D(strides=(2, 2)),
        Dropout(0.5),
        Conv2D(128, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
        Conv2D(256, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
        MaxPooling2D(strides=(2, 2)),
        Dropout(0.5),
        Conv2D(512, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
        Conv2D(512, 3, 3, input_shape=(32, 16, 1), border_mode='same', activation='relu'),
        MaxPooling2D(strides=(2, 2)),
        Dropout(0.5),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, name='output'),
    ])
    model.compile(optimizer='adam', loss='mse')

    ##
    # Train
    ##

    history = model.fit_generator(gen_batches(X_train, y_train, FLAGS.batch_size),
                                  len(X_train),
                                  FLAGS.num_epochs,
                                  validation_data=gen_batches(X_val, y_val, FLAGS.batch_size),
                                  nb_val_samples=len(X_val))


if __name__ == '__main__':
    tf.app.run()
