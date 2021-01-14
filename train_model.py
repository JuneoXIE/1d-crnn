import tensorflow as tf
import argparse
import os
import numpy as np
import matplotlib as plt
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from crnn_1d_model import crnn_1d, DataGenerator
import keras.backend.tensorflow_backend as KTF

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def get_args():
    parser = argparse.ArgumentParser(description='Training CRNN model')
    parser.add_argument('--data_path', default='C:/Users/june1/Downloads/ECG毕设/dataset/')
    parser.add_argument('--label_file_path', default='C:/Users/june1/Downloads/ECG毕设/dataset/REFERENCE.csv')
    parser.add_argument('--checkpoint_path', default= 'D:/Programming workspaces/pyCharm workspace/CRNN_ECG/checkpoints/best_model.h5')
    parser.add_argument('--n_classes', default=2, type=int)
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--test_batch_size', default=25, type=int)
    parser.add_argument('--feature_length', default=3000, type=int)
    parser.add_argument('--shuffle', default='False')
    parser.add_argument('--processing', default='True')
    args = parser.parse_args()
    return args

def show_acc(history):
    """ 绘制精度曲线 """
    plt.clf()
    history_dict = history.history
    acc = history_dict['binary_accuracy']
    val_acc = history_dict['val_binary_accuracy']

    epochs = range(1, len(val_acc) + 1)

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()

    plt.show()


def train_model(args):
    model = crnn_1d(args.feature_length, args.n_classes)
    training_generator = DataGenerator(args.data_path, args.label_file_path, args.batch_size,
                                       args.feature_length, args.shuffle, args.processing, mode = 'Train')
    testing_generator = DataGenerator(args.data_path, args.label_file_path, args.test_batch_size,
                                       args.feature_length, args.shuffle, args.processing, mode = 'Test')

    print("=> Start training <=")
    checkpoint = ModelCheckpoint(args.checkpoint_path,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')
    history = model.fit_generator(training_generator,
                        epochs = 300,
                        validation_data=testing_generator,
                        callbacks=[checkpoint])
    show_acc(history)

if __name__ == "__main__":
    args = get_args()
    train_model(args)