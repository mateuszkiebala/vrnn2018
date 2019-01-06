import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split
from common.constants import dataset_path
import scipy.misc

class Dataset:
    def __init__(self):
        self._data = None

    def load(self, shuffle=True):
        boards_path, results_path = dataset_path()
        x, y = np.load(boards_path), np.load(results_path)
        if shuffle:
            x, y = utils.shuffle(x, y, random_state=42)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        self._data = ((x_train.astype('float32'), y_train), (x_test.astype('float32'), y_test))

    def data(self, type=''):
        if type == 'split':
            (x_train, y_train), (x_test, y_test) = self._data
            return (x_train[:,0], x_train[:,1], y_train), (x_test[:,0], x_test[:,1], y_test)
        if type == 'concat':
            (x_train, y_train), (x_test, y_test) = self._data
            scipy.misc.imsave('outfile100.jpg', x_train[10])
            print(y_train[10])
            #train = (x_train[:, 2], y_train)
            #test = (x_test[:, 2], y_test)
            #train = (np.concatenate((x_train[:,0], x_train[:,1]), axis=2), y_train)
            #test = (np.concatenate((x_test[:,0], x_test[:,1]), axis=2), y_test)

            return self._data
        return self._data

    def triples_data(self):
        (x_train, y_train), (x_test, y_test) = self._data
        #scipy.misc.imsave('outfile100.jpg', x_train[10][0])
        #scipy.misc.imsave('outfile101.jpg', x_train[10][1])
        #scipy.misc.imsave('outfile102.jpg', x_train[10][2])
        #print(y_train[10])
        #scipy.misc.imsave('outfile110.jpg', x_train[11][0])
        #scipy.misc.imsave('outfile111.jpg', x_train[11][1])
        #scipy.misc.imsave('outfile112.jpg', x_train[11][2])
        #print(y_train[11])
        return (x_train[:,0], x_train[:,1], x_train[:,2], y_train), (x_test[:,0], x_test[:,1], x_test[:,2], y_test)




