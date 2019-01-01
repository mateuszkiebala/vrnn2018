import numpy as np
from sklearn import utils
from sklearn.model_selection import train_test_split
from common.constants import *

class Dataset:
    def load(shuffle=True):
        boards_path, results_path = dataset_path()
        x, y = np.load(boards_path), np.load(results_path)
        if shuffle:
            x, y = utils.shuffle(x, y, random_state=42)

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        return (x_train.astype('float32'), y_train), (x_test.astype('float32'), y_test)



