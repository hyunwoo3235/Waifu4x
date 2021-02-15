from abc import ABC

import tensorflow as tf
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing import image
import numpy as np
import os


class Dataset(Sequence, ABC):
    def __init__(self,
                 path: str,
                 batch_size: int,
                 shuffle: bool = True
                 ):
        self.path = path
        self.x_list = os.listdir(self.path)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.x_list)

    def __len__(self):
        return int(np.floor(len(self.x_list) / self.batch_size))

    def __getitem__(self, item):
        return "a"

