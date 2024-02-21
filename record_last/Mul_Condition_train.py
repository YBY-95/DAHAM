import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import glob
from mcCNN import DCNN, FCNN, ori_FCNN, MulDCNN
from dataset import make_sign_dataset

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def train(train_name):

    ## 直接添加工况信息的DANN

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    epochs = 8000  # 训练步数
    batch_size = 32  # batch size
    test_batch_size = 32
    learning_rate = 0.01
    is_training = True
    class_num = 6
    dataset_name = 'CWRU'
    train_name = 'parallel'
    # 获取数据集路径

    print(train_name)
    path = 'D:/DATABASE/CWRU_make_samples/samples/mixes/**.mat'
    signal_path = glob.glob(path)
    print('source domain signals num:', len(signal_path))

    # 构建数据集对象
    train_set, test_set, _, data, train_num, test_num = make_sign_dataset(signal_path, batch_size, dataset_name=dataset_name)
    print('train_set num', train_num)
    print('test_set num', test_num)
    print('sample shape [4096,1]')
    train_set = train_set.repeat()  # 重复循环
    train_iter = iter(train_set)
    test_set = test_set.repeat()
    test_iter = iter(test_set)

    if train_name == 'series':
        # 创建特征提取网络
        feature = DCNN()
        classifer = FCNN()
        feature.build(input_shape=(batch_size, 4096, 1))
        classifer.build(input_shape=(batch_size, 9216))

        # 创建原始的CNN网络
        ori_feature = DCNN()
        ori_classifer = ori_FCNN()
        ori_feature.build(input_shape=(batch_size, 4096, 1))
        ori_classifer.build(input_shape=(batch_size, 8192))


        # 创建优化器
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        optimizer2 = keras.optimizers.SGD(learning_rate=learning_rate)

        #  创建tensorboard记录器
        train_log_dir = 'D:/python_workfile/Multi-condition diagnosis/tensorboard/' + train_name + '/train'
        test_log_dir = 'D:/python_workfile/Multi-condition diagnosis/tensorboard/' + train_name + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(epochs):  # 训练epochs次
            # 2. 训练分类器
            batch_x = next(train_iter)
            sign = batch_x[0]
            load = batch_x[2]
            real_label = tf.one_hot(batch_x[1], depth=class_num)
            # 分类器前向计算
            with tf.GradientTape(persistent=True) as tape:
                extract_feature = feature.call(sign, is_training)
                load_feature = tf.concat([extract_feature, tf.transpose(tf.cast(load, tf.float32)*tf.transpose(
                    tf.ones([batch_size, 1024])))], 1)
                pred_label = classifer.call(load_feature, is_training)
                crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
                y_loss = crossentro(real_label, pred_label)

            correct_train = np.sum(tf.cast(
                tf.equal(
                    tf.one_hot(tf.argmax(pred_label, axis=-1), depth=class_num),
                    tf.cast(real_label, tf.float32)),
                tf.int64))
            correct_train = (correct_train - batch_size*(class_num-2))/2

            grad1 = tape.gradient(y_loss, feature.trainable_variables)
            optimizer.apply_gradients(zip(grad1, feature.trainable_variables))
            grad2 = tape.gradient(y_loss, classifer.trainable_variables)
            optimizer.apply_gradients(zip(grad2, classifer.trainable_variables))

            with tf.GradientTape(persistent=True) as tape2:
                ori_extract_feature = ori_feature.call(sign, is_training)
                ori_pred_label = ori_classifer.call(ori_extract_feature, is_training)
                ori_crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
                ori_loss = ori_crossentro(real_label, ori_pred_label)

            ori_correct_train = np.sum(tf.cast(
                tf.equal(
                    tf.one_hot(tf.argmax(ori_pred_label, axis=-1), depth=class_num),
                    tf.cast(real_label, tf.float32)),
                tf.int64))
            ori_correct_train = (ori_correct_train - batch_size * (class_num - 2)) / 2

            grad3 = tape2.gradient(ori_loss, ori_feature.trainable_variables)
            optimizer2.apply_gradients(zip(grad3, ori_feature.trainable_variables))
            grad4 = tape2.gradient(ori_loss, ori_classifer.trainable_variables)
            optimizer2.apply_gradients(zip(grad4, ori_classifer.trainable_variables))

            # 记录训练标量
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', y_loss, step=epoch)
                tf.summary.scalar('train_acc', correct_train / batch_size, step=epoch)
                tf.summary.scalar('ori_train_loss', ori_loss, step=epoch)
                tf.summary.scalar('ori_train_acc', ori_correct_train / batch_size, step=epoch)

            # 可视化 损失 准确率
            if epoch % 100 == 0:
                correct_test = 0
                ori_correct_test = 0
                total = 0
                test_loss = []
                ori_test_loss = []

                for x in range(test_num//test_batch_size):
                    batch_t = next(test_iter)
                    sign_test = batch_t[0]
                    label_test = tf.one_hot(batch_t[1], depth=class_num)
                    load_test = batch_t[2]

                    extract_feature = feature.call(sign_test, training=None)
                    load_feature = tf.concat([extract_feature, tf.transpose(
                        tf.cast(load_test, tf.float32) * tf.transpose(tf.ones([test_batch_size, 1024])))], 1)
                    pred_label = classifer.call(load_feature, training=None)
                    ori_pred_label = ori_classifer.call(ori_feature.call(sign_test, training=None), training=None)

                    correct = np.sum(tf.cast(
                        tf.equal(
                            tf.one_hot(tf.argmax(pred_label, axis=-1), depth=class_num),
                            tf.cast(label_test, tf.float32)),
                        tf.int64))
                    correct_test += (correct - test_batch_size * (class_num-2)) / 2
                    test_loss.append(crossentro(label_test, pred_label))

                    ori_correct = np.sum(tf.cast(
                        tf.equal(
                            tf.one_hot(tf.argmax(ori_pred_label, axis=-1), depth=class_num),
                            tf.cast(label_test, tf.float32)),
                        tf.int64))
                    ori_correct_test += (ori_correct - test_batch_size * (class_num - 2)) / 2
                    ori_test_loss.append(crossentro(label_test, pred_label))
                    # 此处结果的对比有一些不科学，预测输出的各个标签的概率值，取其中最大值去比较
                    total += test_batch_size

                t_loss = tf.reduce_mean(test_loss)
                ori_t_loss = tf.reduce_mean(ori_test_loss)
                # 绘制混淆矩阵
                # if epoch % 1000 == 0:
                #     plot_confusion_matrix(batch_x[1], tf.argmax(feature.call(sign_test, training=None), axis=-1), class_num)

                print(epoch, '--', 'mul_con'
                      'train_loss:', round(float(y_loss), 4),
                      'test_loss', round(float(t_loss), 4),
                      'train_acc', round(correct_train/batch_size, 4),
                      'test_acc', round(correct_test / total, 4),
                      '------------',
                      'ori', 'train_loss', round(float(ori_loss), 4),
                      'test_loss', round(float(ori_t_loss), 4),
                      'train_acc', round(ori_correct_train/batch_size, 4),
                      'test_acc', round(correct_test/total, 4))

                with test_summary_writer.as_default():
                    tf.summary.scalar('test_loss', t_loss, step=epoch)
                    tf.summary.scalar('test_acc', correct_test / total, step=epoch)
                    tf.summary.scalar('ori_test_loss', ori_t_loss, step=epoch)
                    tf.summary.scalar('ori_test_acc', ori_correct_test / total, step=epoch)

    elif train_name == 'parallel':
        # 创建特征提取网络
        feature = MulDCNN()
        classifer = FCNN()
        feature.build(input_shape=(batch_size, 4096, 2, 1))
        classifer.build(input_shape=(batch_size, 8192))

        # 创建原始的CNN网络
        ori_feature = DCNN()
        ori_classifer = ori_FCNN()
        ori_feature.build(input_shape=(batch_size, 4096, 1))
        ori_classifer.build(input_shape=(batch_size, 8192))

        # 创建优化器
        optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        optimizer2 = keras.optimizers.SGD(learning_rate=learning_rate)

        #  创建tensorboard记录器
        train_log_dir = 'D:/python_workfile/Multi-condition diagnosis/tensorboard/' + train_name + '/train'
        test_log_dir = 'D:/python_workfile/Multi-condition diagnosis/tensorboard/' + train_name + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

        for epoch in range(epochs):  # 训练epochs次
            # 2. 训练分类器
            batch_x = next(train_iter)
            sign = batch_x[0]
            load = batch_x[2]
            real_label = tf.one_hot(batch_x[1], depth=class_num)
            # 分类器前向计算
            with tf.GradientTape(persistent=True) as tape:
                sign_p = tf.stack([tf.cast(sign, tf.float32), tf.transpose(tf.cast(load, tf.float32) * tf.transpose(
                    tf.ones([batch_size, 4096, 1])))], 2)
                extract_feature = feature.call(sign_p, is_training)
                pred_label = classifer.call(extract_feature, is_training)
                crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
                y_loss = crossentro(real_label, pred_label)

            correct_train = np.sum(tf.cast(
                tf.equal(
                    tf.one_hot(tf.argmax(pred_label, axis=-1), depth=class_num),
                    tf.cast(real_label, tf.float32)),
                tf.int64))
            correct_train = (correct_train - batch_size * (class_num - 2)) / 2

            grad1 = tape.gradient(y_loss, feature.trainable_variables)
            optimizer.apply_gradients(zip(grad1, feature.trainable_variables))
            grad2 = tape.gradient(y_loss, classifer.trainable_variables)
            optimizer.apply_gradients(zip(grad2, classifer.trainable_variables))

            with tf.GradientTape(persistent=True) as tape2:
                ori_extract_feature = ori_feature.call(sign, is_training)
                ori_pred_label = ori_classifer.call(ori_extract_feature, is_training)
                ori_crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
                ori_loss = ori_crossentro(real_label, ori_pred_label)

            ori_correct_train = np.sum(tf.cast(
                tf.equal(
                    tf.one_hot(tf.argmax(ori_pred_label, axis=-1), depth=class_num),
                    tf.cast(real_label, tf.float32)),
                tf.int64))
            ori_correct_train = (ori_correct_train - batch_size * (class_num - 2)) / 2

            grad3 = tape2.gradient(ori_loss, ori_feature.trainable_variables)
            optimizer2.apply_gradients(zip(grad3, ori_feature.trainable_variables))
            grad4 = tape2.gradient(ori_loss, ori_classifer.trainable_variables)
            optimizer2.apply_gradients(zip(grad4, ori_classifer.trainable_variables))

            # 记录训练标量
            with train_summary_writer.as_default():
                tf.summary.scalar('train_loss', y_loss, step=epoch)
                tf.summary.scalar('train_acc', correct_train / batch_size, step=epoch)
                tf.summary.scalar('ori_train_loss', ori_loss, step=epoch)
                tf.summary.scalar('ori_train_acc', ori_correct_train / batch_size, step=epoch)

            # 可视化 损失 准确率
            if epoch % 100 == 0:
                correct_test = 0
                ori_correct_test = 0
                total = 0
                test_loss = []
                ori_test_loss = []

                for x in range(test_num // test_batch_size):
                    batch_t = next(test_iter)
                    sign_test = batch_t[0]
                    label_test = tf.one_hot(batch_t[1], depth=class_num)
                    load_test = batch_t[2]

                    sign_test_p = tf.stack([tf.cast(sign_test, tf.float32), tf.transpose(tf.cast(load_test, tf.float32) * tf.transpose(
                        tf.ones([batch_size, 4096, 1])))], 2)
                    extract_feature = feature.call(sign_test_p, training=None)
                    pred_label = classifer.call(extract_feature, training=None)
                    ori_pred_label = ori_classifer.call(ori_feature.call(sign_test, training=None), training=None)

                    correct = np.sum(tf.cast(
                        tf.equal(
                            tf.one_hot(tf.argmax(pred_label, axis=-1), depth=class_num),
                            tf.cast(label_test, tf.float32)),
                        tf.int64))
                    correct_test += (correct - test_batch_size * (class_num - 2)) / 2
                    test_loss.append(crossentro(label_test, pred_label))

                    ori_correct = np.sum(tf.cast(
                        tf.equal(
                            tf.one_hot(tf.argmax(ori_pred_label, axis=-1), depth=class_num),
                            tf.cast(label_test, tf.float32)),
                        tf.int64))
                    ori_correct_test += (ori_correct - test_batch_size * (class_num - 2)) / 2
                    ori_test_loss.append(crossentro(label_test, pred_label))
                    # 此处结果的对比有一些不科学，预测输出的各个标签的概率值，取其中最大值去比较
                    total += test_batch_size

                t_loss = tf.reduce_mean(test_loss)
                ori_t_loss = tf.reduce_mean(ori_test_loss)
                # 绘制混淆矩阵
                # if epoch % 1000 == 0:
                #     plot_confusion_matrix(batch_x[1], tf.argmax(feature.call(sign_test, training=None), axis=-1), class_num)

                print(epoch, '--', 'mul_con'
                                   'train_loss:', round(float(y_loss), 4),
                      'test_loss', round(float(t_loss), 4),
                      'train_acc', round(correct_train / batch_size, 4),
                      'test_acc', round(correct_test / total, 4),
                      '------------',
                      'ori', 'train_loss', round(float(ori_loss), 4),
                      'test_loss', round(float(ori_t_loss), 4),
                      'train_acc', round(ori_correct_train / batch_size, 4),
                      'test_acc', round(ori_correct_test / total, 4))

                with test_summary_writer.as_default():
                    tf.summary.scalar('test_loss', t_loss, step=epoch)
                    tf.summary.scalar('test_acc', correct_test / total, step=epoch)
                    tf.summary.scalar('ori_test_loss', ori_t_loss, step=epoch)
                    tf.summary.scalar('ori_test_acc', ori_correct_test / total, step=epoch)


def main():
    train('0')


if __name__ == '__main__':
    main()