from keras.models import Sequential
from keras.layers import Input, Dense, Conv2D, Flatten
from keras.optimizers import Adam
import tensorflow as tf

def KungFu(self):
    model = Sequential()
    model.add(Input((84,84,4)))
    model.add(Conv2D(filters=32, kernel_size=(8,8), strides=4, activation='relu',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Conv2D(filters=64, kernel_size=(4,4), strides=2, activation='relu',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=1, activation='relu',
                     kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu',
                    kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2)))
    model.add(Dense(len(self.possible_actions), activation='linear'))

    optimizer = Adam(learning_rate=self.learn_rate)
    model.compile(optimizer, loss=tf.keras.losses.Huber())
    return model
