import keras

TESTING = False
batch_size = 15
epochs = 100
body_part = 'class'
opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)