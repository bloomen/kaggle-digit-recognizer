import os
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

np.random.seed(42)

IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNGELS = 1 # grayscale


class Reshaper(TransformerMixin):

    def transform(self, X, *_):
        X = X.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNGELS))
        return X

    def fit(self, *_):
        return self


class AlexNet(BaseEstimator):

    def __init__(self, index):
        self._path = 'output/alexnet_{}.h5'.format(index)

    def fit(self, X, y):
        model = self._create_model()
        y = pd.get_dummies(y).values
        model.fit(X, y, batch_size=64, epochs=3, verbose=1, validation_split=0.1, shuffle=True)
        model.save(self._path)

    def predict(self, X):
        model = keras.models.load_model(self._path)
        y = model.predict(X)
        return np.argmax(y, axis=1)

    def _create_model(self):
        # (3) Create a sequential model
        model = Sequential()

        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=(28,28,1), kernel_size=(3,3),
                         strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling 
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation before passing it to the next layer
        model.add(BatchNormalization())

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(2,2), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(2,2), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(2,2), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Passing it to a dense layer
        model.add(Flatten())
        # 1st Dense Layer
        model.add(Dense(4096, input_shape=(256,)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 2nd Dense Layer
        model.add(Dense(4096))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # 3rd Dense Layer
        model.add(Dense(1000))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))
        # Batch Normalisation
        model.add(BatchNormalization())

        # Output Layer
        model.add(Dense(NUM_LABELS))
        model.add(Activation('softmax'))

        model.summary()

        # (4) Compile 
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model


class CopyTransformer(TransformerMixin):

    def transform(self, df, *_):
        df = pd.DataFrame(df, copy=True)
        return df

    def fit(self, *_):
        return self


def train_data():
    df = pd.read_csv('input/train.csv', dtype=np.float32)
    y = df['label']
    X = df.drop(['label'], axis=1)
    return X.values, y.values


def test_data():
    df = pd.read_csv('input/test.csv', dtype=np.float32)
    return df.values


def make_model():
    return Pipeline([
        ('reshaper', Reshaper()),
        ('model', VotingClassifier([("model_{}".format(i), AlexNet(i)) for i in range(20)])),
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
