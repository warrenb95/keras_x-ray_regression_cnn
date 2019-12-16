import keras

TESTING = False
batch_size = 25
epochs = 100
classification_opt = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
regression_opt = keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)