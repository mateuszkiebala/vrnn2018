import os
import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split
from common.constants import dataset_path, dataset_number_path, GAMES_ARR_PATH

class DataFetcher:
    def __init__(self):
        self.dirs = [dir for dir in os.listdir(GAMES_ARR_PATH) if os.path.isdir(os.path.join(GAMES_ARR_PATH, dir))]

    def fetch_inf(self, type='stack'):
        while True:
            for dir in self.dirs:
                dataset = Dataset()
                dataset.load(number=dir)

                yield dataset.data(type=type)

class Dataset:
    def __init__(self):
        self._data = None
        self._splited = False

    def load(self, number=None, split=True, shuffle=True):
        if number is not None:
            boards_path, results_path = dataset_number_path(number)
            print("Loading board. path:{}, number {}".format(boards_path, str(number)))
        else:
            boards_path, results_path = dataset_path()

        x, y = np.load(boards_path), np.load(results_path)
        if shuffle:
            x, y = utils.shuffle(x, y, random_state=42)

        self._splited = split
        if split:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
            self._data = ((x_train.astype('float32'), y_train), (x_test.astype('float32'), y_test))
        else:
            self._data = (x.astype('float32'), y)

    def data(self, type=''):
        if type == 'split':
            if self._splited:
                (x_train, y_train), (x_test, y_test) = self._data
                return (x_train[:,0], x_train[:,1], y_train), (x_test[:,0], x_test[:,1], y_test)
            else:
                (x, y) = self._data
                return (x[:,0], x[:,1], y)
        if type == 'concat':
            if self._splited:
                (x_train, y_train), (x_test, y_test) = self._data
                train = (np.concatenate((x_train[:,0], x_train[:,1]), axis=2), y_train)
                test = (np.concatenate((x_test[:,0], x_test[:,1]), axis=2), y_test)
                return train, test
            else:
                (x, y) = self._data
                return (np.concatenate((x[:,0], x[:,1]), axis=2), y)

        if type == 'stack':
            if self._splited:
                (x_train, y_train), (x_test, y_test) = self._data
                train = (np.squeeze(np.squeeze(np.stack((x_train[:,0], x_train[:,1]), axis=-1))), y_train)
                test = (np.squeeze(np.squeeze(np.stack((x_test[:,0], x_test[:,1]), axis=-1))), y_test)
                return train, test
            else:
                (x, y) = self._data
                return (np.squeeze(np.squeeze(np.stack((x[:,0], x[:,1]), axis=-1))), y)

        return self._data
