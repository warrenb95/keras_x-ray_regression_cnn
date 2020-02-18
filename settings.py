import tensorflow

TESTING = False
batch_size = 15
epochs = 100
classification_opt = tensorflow.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
regression_opt = tensorflow.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)