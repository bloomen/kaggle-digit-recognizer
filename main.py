import pylab as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import label
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split, GridSearchCV
from sklearn.svm import SVC, LinearSVC, SVR
from sklearn.preprocessing.data import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.preprocessing.data import PolynomialFeatures
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.utils import assert_all_finite
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from featureimpact import FeatureImpact, averaged_impact
import argparse
import pickle
import tensorflow as tf
from sklearn.utils import shuffle

np.random.seed(42)


IMAGE_SIZE = 28
NUM_LABELS = 10
NUM_CHANNGELS = 1 # grayscale


class Reshaper(TransformerMixin):

    def transform(self, X, *_):
        X = X.reshape((-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNGELS))
        X = np.pad(X, ((0, 0),(2, 2),(2, 2),(0, 0)), 'constant')
        return X

    def fit(self, *_):
        return self


class LeNet(BaseEstimator):

    EPOCHS = 10
    BATCH_SIZE = 128
    PATH = 'output/lenet.ckpt'

    def __init__(self):
        self._X = tf.compat.v1.placeholder(tf.float32, shape=[None, 32, 32, 1])
        self._y = tf.compat.v1.placeholder(tf.int32)
        #Invoke LeNet function by passing features
        self._logits = le_net_5(self._X)
        #Softmax with cost function implementation
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self._y, logits=self._logits)
        loss_operation = tf.reduce_mean(cross_entropy)
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001)
        self._training_operation = optimizer.minimize(loss_operation)
        correct_prediction = tf.equal(tf.argmax(self._logits, 1), tf.argmax(self._y, 1))
        self._accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def fit(self, X, y):
        y = pd.get_dummies(y).values
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            num_examples = len(X_train)
            print("Training... with dataset - ", num_examples)
            print()
            for i in range(self.EPOCHS):
                X_train, y_train = shuffle(X_train, y_train)
                for offset in range(0, num_examples, self.BATCH_SIZE):
                    end = offset + self.BATCH_SIZE
                    batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                    sess.run(self._training_operation, feed_dict={self._X: batch_x, self._y: batch_y})
                validation_accuracy = self._evaluate(sess, X_val, y_val)
                print("EPOCH {} ...".format(i+1))
                print("Validation Accuracy = {:.3f}".format(validation_accuracy))
                print()
            saver = tf.compat.v1.train.Saver()
            saver.save(sess, self.PATH)
            print("Model saved {}".format(self.PATH))

    def predict(self, X):
        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.Saver()
            saver.restore(sess, self.PATH)
            print("Model restored {}".format(self.PATH))
            Z = self._logits.eval(feed_dict={self._X: X})
            return np.argmax(Z, axis=1)
 
    def _evaluate(self, sess, X_data, y_data):
        num_examples = len(X_data)
        total_accuracy = 0
        for offset in range(0, num_examples, self.BATCH_SIZE):
            batch_x, batch_y = X_data[offset:offset+self.BATCH_SIZE], y_data[offset:offset+self.BATCH_SIZE]
            accuracy = sess.run(self._accuracy_operation, feed_dict={self._X: batch_x, self._y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        return total_accuracy / num_examples


# LeNet-5 architecture implementation using TensorFlow
def le_net_5(x):
     # Layer 1 : Convolutional Layer. Input = 32x32x1, Output = 28x28x1.
     conv1_w = tf.Variable(tf.random.truncated_normal(shape = [5,5,1,6],mean = 0, stddev = 0.1))
     conv1_b = tf.Variable(tf.zeros(6))
     conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b 
     # TODO: Activation.
     conv1 = tf.nn.relu(conv1)
     
     # Pooling Layer. Input = 28x28x1. Output = 14x14x6.
     pool_1 = tf.nn.max_pool2d(conv1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
     
     # TODO: Layer 2: Convolutional. Output = 10x10x16.
     conv2_w = tf.Variable(tf.random.truncated_normal(shape = [5,5,6,16], mean = 0, stddev = 0.1))
     conv2_b = tf.Variable(tf.zeros(16))
     conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
     # TODO: Activation.
     conv2 = tf.nn.relu(conv2)
    # TODO: Pooling. Input = 10x10x16. Output = 5x5x16.
     pool_2 = tf.nn.max_pool2d(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')
     
     # TODO: Flatten. Input = 5x5x16. Output = 400.
     fc1 = tf.compat.v1.layers.flatten(pool_2)
     
     # TODO: Layer 3: Fully Connected. Input = 400. Output = 120.
     fc1_w = tf.Variable(tf.random.truncated_normal(shape = (400,120), mean = 0, stddev = 0.1))
     fc1_b = tf.Variable(tf.zeros(120))
     fc1 = tf.matmul(fc1,fc1_w) + fc1_b
     
     # TODO: Activation.
     fc1 = tf.nn.relu(fc1)
     
     # TODO: Layer 4: Fully Connected. Input = 120. Output = 84.
     fc2_w = tf.Variable(tf.random.truncated_normal(shape = (120,84), mean = 0, stddev = 0.1))
     fc2_b = tf.Variable(tf.zeros(84))
     fc2 = tf.matmul(fc1,fc2_w) + fc2_b
     # TODO: Activation.
     fc2 = tf.nn.relu(fc2)
     
     # TODO: Layer 5: Fully Connected. Input = 84. Output = 10.
     fc3_w = tf.Variable(tf.random.truncated_normal(shape = (84,10), mean = 0 , stddev = 0.1))
     fc3_b = tf.Variable(tf.zeros(10))
     logits = tf.matmul(fc2, fc3_w) + fc3_b
     return logits


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
        ('model', LeNet()),
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
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(y_pred.shape, y_test.shape)
        print("Accuracy:", accuracy_score(y_test, y_pred))
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
