import tensorflow as tf
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Conv2D, ConvLSTM2D, Dense, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import os

from utils import gen_batches

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_path', 'data/mjc+mrg/combined_hal.csv', 'The path to the csv of training data.')
flags.DEFINE_string('save_dir', 'save/hal_color_2/', 'The directory to which to save the model.')
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

    ##
    # Define Model
    ##

    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(32, 128, 3), padding='same', activation='relu'),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),
        # Conv2D(1024, (3, 3), padding='same', activation='relu'),
        # Conv2D(1024, (3, 3), padding='same', activation='relu'),
        # MaxPooling2D(),
        # Dropout(0.5),
        # Conv2D(2048, (3, 3), padding='same', activation='relu'),
        # Conv2D(2048, (3, 3), padding='same', activation='relu'),
        # MaxPooling2D(),
        # Dropout(0.5),
        Flatten(),
        Dense(1024, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(1, name='output', activation='tanh'),
    ])
    model.compile(optimizer=Adam(lr=FLAGS.lrate), loss='mse')

    ##
    # Train
    ##

    history = model.fit_generator(gen_batches(X_train, y_train, FLAGS.batch_size),
                                  len(X_train) / FLAGS.batch_size,
                                  FLAGS.num_epochs,
                                  validation_data=gen_batches(X_val, y_val, FLAGS.batch_size),
                                  validation_steps=(len(X_val) / FLAGS.batch_size))

    ##
    # Save model
    ##

    if not os.path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    json = model.to_json()
    model.save_weights(os.path.join(FLAGS.save_dir, 'model.h5'))
    with open(os.path.join(FLAGS.save_dir, 'model.json'), 'w') as f:
        f.write(json)


if __name__ == '__main__':
    tf.app.run()
