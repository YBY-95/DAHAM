import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class DCNN(keras.Model):
    # 建立特征提取网络
    def __init__(self):
        super(DCNN, self).__init__()
        self.conv1 = layers.Conv1D(16, 128, 1, 'same', use_bias=False)
        self.maxpl1 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')
        self.do1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv1D(32, 64, 1, 'same', use_bias=False)
        self.maxpl2 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')

        self.conv3 = layers.Conv1D(64, 16, 1, 'same', use_bias=False)
        self.maxpl3 = layers.MaxPool1D(pool_size=2, strides=None, padding='valid')
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


class MulDCNN(keras.Model):
    # 建立特征提取网络
    def __init__(self):
        super(MulDCNN, self).__init__()
        self.conv1 = layers.Conv2D(16, [128, 2], 1, 'same', use_bias=False)
        self.maxpl1 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')
        self.do1 = layers.Dropout(0.5)

        self.conv2 = layers.Conv2D(32, [64, 2], 1, 'same', use_bias=False)
        self.maxpl2 = layers.MaxPool1D(pool_size=4, strides=None, padding='valid')

        self.conv3 = layers.Conv2D(64, [16, 2], 1, 'same', use_bias=False)
        self.maxpl3 = layers.MaxPool1D(pool_size=2, strides=None, padding='valid')
        self.do3 = layers.Dropout(0.5)

        self.conv4 = layers.Conv2D(128, [3, 2], 1, 'same', use_bias=False)
        self.maxpl4 = layers.MaxPool1D(pool_size=2, strides=None, padding='valid')

        self.conv5 = layers.Conv2D(256, [2, 2], 1, 'same', use_bias=False)
        self.maxpl5 = layers.MaxPool1D(pool_size=2, strides=None, padding='valid')
        self.do5 = layers.Dropout(0.5)

        self.flatten = layers.Flatten()

    def call(self, inputs, training=None):
        x = inputs
        x = self.conv1(x)
        x = tf.nn.relu(self.do1(tf.stack([self.maxpl1(x[:, :, 0, :]), self.maxpl1(x[:, :, 1, :])], 2)))
        x = self.conv2(x)
        x = tf.nn.relu(tf.stack([self.maxpl2(x[:, :, 0, :]), self.maxpl2(x[:, :, 1, :])], 2))
        x = self.conv3(x)
        x = tf.nn.relu(self.do3(tf.stack([self.maxpl3(x[:, :, 0, :]), self.maxpl3(x[:, :, 1, :])], 2)))
        x = self.conv4(x)
        x = tf.nn.relu(tf.stack([self.maxpl4(x[:, :, 0, :]), self.maxpl4(x[:, :, 1, :])], 2))
        x = self.conv5(x)
        x = tf.nn.relu(self.do5(tf.stack([self.maxpl5(x[:, :, 0, :]), self.maxpl5(x[:, :, 1, :])], 2)))
        x = self.flatten(x[:, :, 0, :])

        return x


class FCNN(keras.Model):
    def __init__(self):
        super(FCNN, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(6, activation='softmax')

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class ori_FCNN(keras.Model):
    def __init__(self):
        super(ori_FCNN, self).__init__()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(64, activation='relu')
        self.fc3 = layers.Dense(6, activation='softmax')

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def main():
    dcnn = DCNN()
    fcnn = FCNN()


if __name__ == '__main__':
    main()