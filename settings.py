import keras

TESTING = False
batch_size = 25
epochs = 250
# opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
# opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)