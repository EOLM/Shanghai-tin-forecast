import keras
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM


def get_model():
    model = tf.keras.Sequential()
    model.add(LSTM(2048, input_shape=(8, 1)))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation="relu"))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(128, activation="relu"))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-3,
        amsgrad=False,
        name='Adam',
    )

                  )
    return model
