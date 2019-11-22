import keras

TESTING = True
batch_size = 20
epochs = 10
body_part = 'class'
opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)