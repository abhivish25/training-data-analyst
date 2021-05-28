import os
import os
import time
import tensorflow as tf
import numpy as np
from . import model_definition

#Get data

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# add empty color dimension
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

##################### Multiple GPUs or CPUs ###################
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
###############################################################
    model = model_definition.create_model(input_shape=x_train.shape[1:])
    model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, ),
      loss='sparse_categorical_crossentropy',
      metrics=['sparse_categorical_accuracy'])
start = time.time()
model.fit(
    x_train.astype(np.float32), y_train.astype(np.float32),
    epochs=17,
    steps_per_epoch=60,
    validation_data=(x_test.astype(np.float32), y_test.astype(np.float32)),
    validation_freq=17
)
print("Training time with multiple GPUs: {}".format(time.time() - start))
model.save_weights('./fashion_mnist_mult_gpu.h5', overwrite=True)
