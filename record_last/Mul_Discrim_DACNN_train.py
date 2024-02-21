import tensorflow as tf
from tensorflow import keras
from Mul_DACNN import Gfeature, Gy, Gd
import os
import glob
from dataset import make_sign_dataset
import numpy as np
import sys


def main(domain_name_s, domain_name_t, lamda):

    # 输出写入out.txt文件中
    sys.stdout = open('out.txt', 'a')
    # 设置参数
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    epochs = 30000  # 训练步数上限
    batch_size = 32  # batch size
    learning_rate = 0.001
    is_training = True
    data_name = 'CWRU'

    data_path = r'D:/DATABASE/CWRU_make_samples/samples'
    data_domain_name_t = domain_name_t.replace('+', '-')
    data_domain_name_s = domain_name_s.replace('+', '-')
    data_domain_name_st = data_domain_name_s + '-' + data_domain_name_t
    print('lamda = ', lamda)
    print('source domain:', domain_name_s, '  target domain:', domain_name_t)

    # 获取数据集路径
    signal_s_path = glob.glob(data_path+r'/'+data_domain_name_s+r'/**.mat')
    print('source domain signals num:', len(signal_s_path))
    signal_st_path = glob.glob(data_path+r'/'+data_domain_name_st+r'/**.mat')
    print('source&target domain signals num:', len(signal_st_path))
    signal_t_path = glob.glob(data_path+r'/'+data_domain_name_t+r'/**.mat')
    print('target domain signals num:', len(signal_t_path))

    # 构建源域数据集对象
    data_train_s, data_test_s, _, _, train_num_s, test_num_s = make_sign_dataset(signal_s_path,
                                                                                 batch_size, dataset_name=data_name)
    data_train_s = data_train_s.repeat()
    data_test_s = data_test_s.repeat()
    db_s_train_iter = iter(data_train_s)
    db_s_test_iter = iter(data_test_s)

    # 构建源域&目标域数据集
    data_train_st, data_test_st, _, _, train_num_st, test_num_st = make_sign_dataset(signal_st_path,
                                                                                     batch_size, dataset_name=data_name)
    data_train_st = data_train_st.repeat()
    data_test_st = data_test_st.repeat()
    db_st_train_iter = iter(data_train_st)
    db_st_test_iter = iter(data_test_st)

    # 构建目标域测试集
    _, data_test_t, _, _, train_num_t, test_num_t = make_sign_dataset(signal_t_path,
                                                                                 batch_size, dataset_name=data_name)
    # data_train_t = data_train_t.repeat()
    data_test_t = data_test_t.repeat()
    # db_t_train_iter = iter(data_train_t)
    db_t_test_iter = iter(data_test_t)
    print('dataset OK')

    # 载入预训练网络
    feature = Gfeature()
    feature.build(input_shape=(batch_size, 4096, 1))
    classifer = Gy()
    classifer.build(input_shape=(batch_size, 8192))
    feature.load_weights('./ckpt/'+domain_name_s+'-feature.ckpt')
    classifer.load_weights('./ckpt/'+domain_name_s+'-classifer.ckpt')
    print('load weights OK')
    # 创建域分类器
    discrim = Gd()
    discrim.build(input_shape=(batch_size, 8192))
    print('model building OK')

    # 创建优化器
    f_optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    c_optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
    d_optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

    loss_d = []
    loss_c = []

    for epoch in range(epochs):
        batch_s = next(db_s_train_iter)
        batch_st = next(db_st_train_iter)

        sign_st = batch_st[0]
        sign_s = batch_s[0]

        label_real = tf.one_hot(batch_s[1], depth=6)
        domain_real = tf.one_hot(batch_st[2], depth=2)
        Cate_crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
        with tf.GradientTape(persistent=True) as tape:
            feature_c = feature.call(sign_s, is_training)
            label_pred = classifer.call(feature_c, is_training)
            feature_d = feature.call(sign_st, is_training)
            domain_pred = discrim.call(feature_d, is_training)

            c_loss = Cate_crossentro(label_real, label_pred)
            d_loss = Cate_crossentro(domain_real, domain_pred)
            d1_loss = Cate_crossentro(1-domain_real, domain_pred)
            E_d = c_loss + lamda*d1_loss

        grad_d = tape.gradient(d_loss, discrim.trainable_variables)
        d_optimizer.apply_gradients(zip(grad_d, discrim.trainable_variables))

        grad_c = tape.gradient(c_loss, classifer.trainable_variables)
        c_optimizer.apply_gradients(zip(grad_c, classifer.trainable_variables))

        grad_f = tape.gradient(E_d, feature.trainable_variables)
        f_optimizer.apply_gradients(zip(grad_f, feature.trainable_variables))

        if epoch % 2000 == 0:
            loss_c.append(float(c_loss))
            loss_d.append(float(d_loss))
            correct_s_c = 0
            correct_st_d = 0
            correct_t_c = 0
            total_s_c = 0
            total_st_d = 0
            total_t_c = 0

            for x in range(30):
                batch_s = next(db_s_test_iter)
                batch_st = next(db_st_test_iter)
                batch_t = next(db_t_test_iter)
                sign_test_s = batch_s[0]
                label_test_s = batch_s[1]
                sign_test_st = batch_st[0]
                domain_test_st = batch_st[2]
                sign_test_t = batch_t[0]
                label_test_t = batch_t[1]
                # 源域数据分类正确率
                correct = tf.cast(
                    tf.equal(
                        tf.argmax(classifer.call(feature.call(sign_test_s, training=None), training=None), axis=-1),
                        tf.cast(label_test_s, tf.int64)), tf.float32)
                correct_s_c += np.sum(correct)
                total_s_c += batch_size
                # 混合数据判别正确率
                correct = tf.cast(
                    tf.equal(
                        tf.argmax(discrim.call(feature.call(sign_test_st, training=None), training=None), axis=-1),
                        tf.cast(domain_test_st, tf.int64)), tf.float32)
                correct_st_d += np.sum(correct)
                total_st_d += batch_size
                # 目标域数据分类正确率
                correct = tf.cast(
                    tf.equal(
                        tf.argmax(classifer.call(feature.call(sign_test_t, training=None), training=None), axis=-1),
                        tf.cast(label_test_t, tf.int64)), tf.float32)
                correct_t_c += np.sum(correct)
                total_t_c += batch_size
            print(epoch,
                  'c-loss:', float(c_loss),
                  'd-loss:', float(d_loss),
                  's_c acc', correct_s_c / total_s_c,
                  'st_d acc', correct_st_d / total_st_d,
                  't_c acc', correct_t_c / total_t_c, )

    feature.save_weights('./ckpt/' + domain_name_s + '-' + domain_name_t + '-feature.ckpt')
    classifer.save_weights('./ckpt/' + domain_name_s + '-' + domain_name_t + '-classifer.ckpt')
    print('weights save OK')
    sys.stdout.close()


def test_otherdomain(test_data_domain, domain_name_s, domain_name_t):

    sys.stdout = open('out.txt', 'a')
    # 用于迁移训练完成之后，对其他域数据的分类效果，判别模型是否提取到了共同特征
    path = r'D:/DATABASE/CWRU_make_samples/samples'
    data_path = glob.glob(path+'/'+test_data_domain+'/**.mat')
    batch_size = 32
    data_name = 'CWRU'
    print('load condition', test_data_domain, 'samples number', len(data_path))

    test_data, _, _, _, test_num, _ = make_sign_dataset(data_path, batch_size, data_name)
    data = test_data.repeat()
    data = iter(data)

    feature = Gfeature()
    feature.build(input_shape=(batch_size, 4096, 1))
    feature.load_weights('./ckpt/' + domain_name_s + '-' + domain_name_t + '-feature.ckpt')
    classifer = Gy()
    classifer.build(input_shape=(batch_size, 8192))
    classifer.load_weights('./ckpt/' + domain_name_s + '-' + domain_name_t + '-classifer.ckpt')
    print('model source domain', domain_name_s, 'target domain', domain_name_t)
    correct_test = 0
    total = 0
    crossentro = tf.keras.losses.CategoricalCrossentropy()

    for x in range(test_num // batch_size):
        batch_x = next(data)
        sign = batch_x[0]
        label_test = tf.one_hot(batch_x[1], depth=6)
        label_pred_test = classifer.call(feature.call(sign, training=None), training=None)
        correct = tf.cast(
            tf.equal(
                tf.argmax(label_pred_test, axis=-1),
                tf.argmax(label_test, axis=-1)), tf.float32)
        correct_test += np.sum(correct)
        total += batch_size
        t_loss = tf.reduce_mean(crossentro(label_test, label_pred_test))

    print('loss', float(t_loss),
          'acc', correct_test / total)
    sys.stdout.close()


if __name__ == '__main__':

    main('0+1+2', '3', 0.01)
