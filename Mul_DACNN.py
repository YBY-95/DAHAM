import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Gfeature(keras.Model):
    # 建立特征提取网络
    def __init__(self):
        super(Gfeature, self).__init__()
        self.conv1 = layers.Conv1D(16, 128, 1, 'same', use_bias=False)
        self.maxpl1 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')
        self.do1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv1D(32, 64, 1, 'same', use_bias=False)
        self.maxpl2 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')

        self.conv3 = layers.Conv1D(64, 16, 1, 'same', use_bias=False)
        self.maxpl3 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')
        self.do3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv1D(128, 3, 1, 'same', use_bias=False)
        self.maxpl4 = layers.MaxPool1D(pool_size=2, strides=None, padding='valid')

        self.conv5 = layers.Conv1D(256, 2, 1, 'same', use_bias=False)
        self.maxpl5 = layers.MaxPool1D(pool_size=2, strides=None, padding='valid')
        self.do5 = layers.Dropout(0.5)

        self.flatten = layers.Flatten()

    def call(self, inputs, training=None):

        x = inputs
        x = tf.nn.relu(self.do1(self.maxpl1(self.conv1(x))))
        x = tf.nn.relu(self.maxpl2(self.conv2(x)))
        x = tf.nn.relu(self.do3(self.maxpl3(self.conv3(x))))
        x = tf.nn.relu(self.maxpl4(self.conv4(x)))
        x = tf.nn.relu(self.do5(self.maxpl5(self.conv5(x))))
        x = self.flatten(x)

        return x


class Gfeature1(keras.Model):
    # 建立特征提取网络
    def __init__(self):
        super(Gfeature1, self).__init__()
        self.conv1 = layers.Conv1D(16, 128, 1, 'same', use_bias=False)
        self.maxpl1 = layers.MaxPool1D(pool_size=8, strides=None, padding='valid')
        self.do1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv1D(32, 64, 1, 'same', use_bias=False)
        self.maxpl2 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')
        self.do2 = layers.Dropout(0.5)

        self.flatten = layers.Flatten()

    def call(self, inputs, training=None):

        x = inputs
        x = tf.nn.relu(self.do1(self.maxpl1(self.conv1(x))))
        x = tf.nn.relu(self.do2(self.maxpl2(self.conv2(x))))
        x = self.flatten(x)

        return x


class Gfeature2(keras.Model):
    # 建立特征提取网络
    def __init__(self):
        super(Gfeature2, self).__init__()
        self.conv1 = layers.Conv1D(16, 128, 1, 'same', use_bias=False)
        self.maxpl1 = layers.MaxPool1D(pool_size=8, strides=None, padding='valid')
        self.do1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv1D(32, 64, 1, 'same', use_bias=False)
        self.maxpl2 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')

        self.conv3 = layers.Conv1D(64, 16, 1, 'same', use_bias=False)
        self.maxpl3 = layers.MaxPool1D(pool_size=2, strides=None, padding='valid')
        self.do3 = layers.Dropout(0.5)

        self.flatten = layers.Flatten()

    def call(self, inputs, training=None):

        x = inputs
        x = tf.nn.relu(self.do1(self.maxpl1(self.conv1(x))))
        x = tf.nn.relu(self.maxpl2(self.conv2(x)))
        x = tf.nn.relu(self.do3(self.maxpl3(self.conv3(x))))
        x = self.flatten(x)

        return x


class Gd(keras.Model):
    # 域分类器
    def __init__(self):
        super(Gd, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(2, activation='softmax')

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc3((self.fc1(self.fc2(x))))

        return x


class GdWD(keras.Model):
    # 域分类器
    def __init__(self):
        super(GdWD, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)
        x1 = self.fc3(x)

        return x1

class GdWD3(keras.Model):
    # 域分类器
    def __init__(self):
        super(GdWD3, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(1)

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)

        return x


class Gy(keras.Model):
    # 分类器
    def __init__(self):
        super(Gy, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(8, activation='softmax')

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def main():
    gfeature = Gfeature()
    gd = Gd()
    yclassifer = Gy()


if __name__ == '__main__':
    main()