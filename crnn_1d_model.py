from keras.models import Model, load_model, Sequential
from keras.layers import BatchNormalization, Conv1D, Dense, Input, Dropout, Activation, \
    Bidirectional, GRU, MaxPool1D, LeakyReLU, Flatten
import math
import numpy as np
from scipy.io import loadmat
from scipy import signal
import pandas as pd
import os
import random
from keras.utils import Sequence, to_categorical


# 创建CRNN模型
def crnn_1d(feature_length, n_classes):
    print('=> Building model <=')
    # 输入层
    model = Sequential()
    # 第一个CONV层
    model.add(Conv1D(64, kernel_size = 9, activation='relu',
                       name='conv1d_1', input_shape=(feature_length,1)))
    model.add(MaxPool1D(pool_size = 6, strides = 2))
    model.add(LeakyReLU(alpha=0.5))

    # 第二个CONV层
    model.add(Conv1D(128, kernel_size = 12, activation='relu',
                       name='conv_1d_2'))
    model.add(MaxPool1D(pool_size = 9, strides = 3))

    # Dropout层
    model.add(Dropout(0.5))

    # batch nomalization层
    model.add(BatchNormalization(name='bn_conv_1d'))

    # Bi GRU层
    model.add(Bidirectional(GRU(64, return_sequences=True, name='bi_gru1')))

    # Dropout层
    model.add(Dropout(0.5))

    # 全连接层
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['binary_accuracy'])
    print(model.summary())
    print("=> Build completed! <=")

    return model


class DataGenerator(Sequence):
    def __init__(self, dataset_path, label_file_path, batch_size = 25, feature_length = 3000, shuffle=False, processing = True, mode = 'Train'):
        'Initialization'
        self.dataset_path = dataset_path
        self.feature_length = feature_length
        self.batch_size = batch_size
        self.n_classes = 2
        self.shuffle = shuffle
        self.processing = processing
        self.diease = 2
        self.mode = mode
        if self.mode == 'Train':
            self.start = 0
            self.stop = 4125
        elif self.mode == 'Test':
            self.start = 4125
            self.stop = 5500
        else:
            self.start = 5500
            self.stop = 6877

        self.labels = pd.read_csv(label_file_path)
        self.labels = self.labels.reset_index(drop=True)
        self.samples = list(filter(lambda filename: os.path.splitext(filename)[1] == '.mat', os.listdir(self.dataset_path)))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int((self.stop - self.start) / self.batch_size)

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        # Generate indexes of the batch
        sample_list = [self.samples[x] for x in range(self.start + batch_index * self.batch_size,
                                         self.start + (batch_index + 1) * self.batch_size)]

        X = np.zeros((len(sample_list), self.feature_length, 1))
        y = np.empty((len(sample_list)), dtype=int)

        for index, sample in enumerate(sample_list):
            # Record data
            file_path = os.path.join(self.dataset_path, sample)
            wave = np.array(loadmat(file_path)['ECG'])[0][0][2][1][0:3000]
            if self.processing:
                b, a = signal.butter(5, [2 * 0.1 / 500, 2 * 50 / 500], 'bandpass')
                wave = signal.lfilter(b, a, wave)
            X[index,:,0] = wave

            # Record label
            sample_index = int(sample[1:5]) - 1
            if self.diease in [self.labels['First_label'][sample_index],
                           self.labels['Second_label'][sample_index],
                           self.labels['Third_label'][sample_index]]:
                y[index] = 1
            else:
                y[index] = 0

        return X, y

    def on_epoch_end(self):
        'Updates after each epoch'
        if self.shuffle:
            random.shuffle(self.samples)