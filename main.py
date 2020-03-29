import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import argparse
import pickle
import pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.utils import assert_all_finite
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import SGD
from googlenet import create_googlenet
from PIL import Image


np.random.seed(42)

IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNELS = 1 # grayscale


class Reshaper(TransformerMixin):

    def transform(self, X, *_):
        X_out = []
        for i, img in enumerate(X):
            img = img.reshape((IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
            img = np.pad(img, ((98, 98),(98, 98),(1, 1)), 'constant')
#            print(img.shape)
#            img = np.array(Image.fromarray(img).resize((224, 224))).astype(np.float32)
            img[:, :, 0] -= 123.68
#            img[:, :, 1] -= 116.779
#            img[:, :, 2] -= 103.939
            img[:,:,[0,1,2]] = img[:,:,[2,1,0]]
            img = img.transpose((2, 0, 1))
            img = np.expand_dims(img, axis=0)
            X_out.append(img)
        return np.array(X_out)

    def fit(self, *_):
        return self


class AlexNet(BaseEstimator):

    def __init__(self):
        self._path = 'output/googlenet.h5'

    def fit(self, X, y):
        y = pd.get_dummies(y).values
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model = self._create_model()
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X, y, batch_size=64, epochs=1, verbose=1, validation_split=0.1, shuffle=True)
        model.save(self._path)

    def predict(self, X):
        model = keras.models.load_model(self._path)
        y = model.predict(X)[2]
        return np.argmax(y, axis=1)

    def _create_model(self):
        # (3) Create a sequential model
        model = create_googlenet()
        return model


class CopyTransformer(TransformerMixin):

    def transform(self, df, *_):
        df = pd.DataFrame(df, copy=True)
        return df

    def fit(self, *_):
        return self


def train_data():
    df = pd.read_csv('input/train.csv', dtype=np.float32).loc[:1000, :]
    y = df['label']
    X = df.drop(['label'], axis=1)
    return X.values, y.values


def test_data():
    df = pd.read_csv('input/test.csv', dtype=np.float32)
    return df.values


def make_model():
    return Pipeline([
        ('reshaper', Reshaper()),
        ('model', AlexNet()),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", help="Ensure good data quality", type=str,
                        choices=['train', 'test'])
    parser.add_argument("--train", help="Train the model", type=str,
                        choices=['cv', 'best'])
    parser.add_argument("--evaluate", help="Evaluate the model", type=str,
                        choices=['cv', 'best'])
    parser.add_argument("--submission", help="Generate submission on test data", type=str,
                        choices=['cv', 'best'])
    args = parser.parse_args()

    try:
        os.mkdir('output')
    except FileExistsError:
        pass

    if args.quality:
        if args.quality == 'train':
            X, y = train_data()
            X = Reshaper().fit_transform(X)
            print(y.shape, y.dtype)
        else: # test
            X = test_data()
            X = Reshaper().fit_transform(X)
        print(X.shape, X.dtype)
        assert_all_finite(X)
        return

    if args.train == 'cv':
        X, y = train_data()
        model = make_model()
        print("Full training...")
        model.fit(X, y)
        return
 
    if args.train == 'best':
        raise RuntimeError("Not implemented")
        return

    if args.submission:
        if args.submission == 'best':
            raise RuntimeError("Not implemented")
        X = test_data()
        model = make_model()
        y_pred = model.predict(X)
        submission = pd.DataFrame({
            'ImageId': list(range(1, len(y_pred) + 1)),
            'Label': y_pred,
        })
        submission.to_csv('output/submission{}.csv'.format(args.submission), index=False)
        return

    if args.evaluate:
        raise RuntimeError("Not implemented")
        return


if __name__ == '__main__':
    main()
