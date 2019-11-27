import keras

TESTING = False
batch_size = 25
epochs = 100
opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
# opt = keras.optimizers.Adam(learning_rate=0.001, amsgrad=True)