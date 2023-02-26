from keras.datasets import mnist
import numpy as np

from RMDL import RMDL_Image as RMDL

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train_D = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test_D = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    X_train = X_train_D / 255.0
    X_test = X_test_D / 255.0
    number_of_classes = np.unique(y_train).shape[0]
    shape = (28, 28, 1)
    batch_size = 128
    sparse_categorical = 0

    n_epochs = [100, 100, 100]  ## DNN--RNN-CNN
    Random_Deep = [0, 0, 3]  ## DNN--RNN-CNN
    RMDL.Image_Classification(X_train, y_train, X_test, y_test, shape,
                              batch_size=batch_size,
                              sparse_categorical=True,
                              random_deep=Random_Deep,
                              epochs=n_epochs)
