import tensorflow as tf
from tensorflow import keras
from Mul_DACNN import Gfeature, Gy, Gd
import os
import glob
from dataset import make_sign_dataset
import numpy as np
import xlwings as xw
import pandas as pd
from mmd_tf import mmd_loss as mdl
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"


def main(learning_rate, domain_name_s, domain_name_t):

    # 输出写入out.txt文件中,同时run界面中的输出关闭
    # sys.stdout = open('out.txt', 'a')

    # 设置参数
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    epochs = 10000# 训练步数上限
    batch_size = 128  # batch size
    is_training = True
    dataset_name = 'CWRU'
    data_name = 'crack_0.007'
    print('crack_size', data_name)

    data_path = r'D:/DATABASE/CWRU_make_samples/samples/'+data_name
    data_domain_name_t = domain_name_t.replace('+', '-')
    data_domain_name_s = domain_name_s.replace('+', '-')
    data_domain_name_st = data_domain_name_s + '-' + data_domain_name_t
    print('source domain:', domain_name_s, '  target domain:', domain_name_t)

    # 获取数据集路径
    signal_s_path = glob.glob(data_path + r'/' + data_domain_name_s + r'/**.mat')
    print('source domain signals num:', len(signal_s_path))
    signal_st_path = glob.glob(data_path + r'/' + data_domain_name_st + r'/**.mat')
    print('source&target domain signals num:', len(signal_st_path))
    signal_t_path = glob.glob(data_path + r'/' + data_domain_name_t + r'/**.mat')
    print('target domain signals num:', len(signal_t_path))

    # 构建源域数据集对象
    data_train_s, data_test_s, _, _, train_num_s, test_num_s = make_sign_dataset(signal_s_path,
                                                                                 batch_size, dataset_name=dataset_name)
    data_train_s = data_train_s.repeat()
    data_test_s = data_test_s.repeat()
    db_s_train_iter = iter(data_train_s)
    db_s_test_iter = iter(data_test_s)

    # 构建源域&目标域数据集
    data_train_st, data_test_st, _, _, train_num_st, test_num_st = make_sign_dataset(signal_st_path,
                                                                                     batch_size, dataset_name=dataset_name)
    data_train_st = data_train_st.repeat()
    data_test_st = data_test_st.repeat()
    db_st_train_iter = iter(data_train_st)
    db_st_test_iter = iter(data_test_st)

    # 构建目标域测试集
    data_train_t, data_test_t, _, _, train_num_t, test_num_t = make_sign_dataset(signal_t_path,
                                                                      batch_size, dataset_name=dataset_name)
    data_train_t = data_train_t.repeat()
    data_test_t = data_test_t.repeat()
    db_t_train_iter = iter(data_train_t)
    db_t_test_iter = iter(data_test_t)
    print('dataset OK')

    # 载入预训练网络
    feature = Gfeature()
    feature.build(input_shape=(batch_size, 4096, 1))
    classifer = Gy()
    classifer.build(input_shape=(batch_size, 8192))
    feature.load_weights('./ckpt/pretrain/' + data_name + '/' + domain_name_s + '-feature.ckpt')
    classifer.load_weights('./ckpt/pretrain/' + data_name + '/' + domain_name_s + '-classifer.ckpt')
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

    for epoch in range(epochs + 1):
        batch_s = next(db_s_train_iter)
        batch_st = next(db_st_train_iter)
        batch_t = next(db_t_train_iter)

        sign_st = batch_st[0]
        sign_s = batch_s[0]
        sign_t = batch_t[0]

        label_real = tf.one_hot(batch_s[1], depth=6)
        domain_real = tf.one_hot(domain_replace(batch_st[2], batch_size, data_domain_name_s, data_domain_name_t), depth=2)
        Cate_crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
        with tf.GradientTape(persistent=True) as tape:
            feature_c = feature.call(sign_s, is_training)
            label_pred = classifer.call(feature_c, is_training)
            feature_d = feature.call(sign_st, is_training)
            domain_pred = discrim.call(feature_d, is_training)
            mmd_loss = mdl(tf.cast(tf.reshape(sign_s, [sign_s.shape[0], sign_s.shape[1]]), tf.float32),
                           tf.cast(tf.reshape(sign_t, [sign_t.shape[0], sign_t.shape[1]]), tf.float32)
                           , 1)

            c_loss = Cate_crossentro(label_real, label_pred)
            d_loss = Cate_crossentro(domain_real, domain_pred)
            d1_loss = Cate_crossentro(1 - domain_real, domain_pred)
            if epoch == 0:
                t_r_weight = 0
            else:
                t_r_weight = 1/(correct_t_c / total_t_c)
            E_d = c_loss + mmd_loss * d1_loss + t_r_weight

        grad_d = tape.gradient(d_loss, discrim.trainable_variables)
        d_optimizer.apply_gradients(zip(grad_d, discrim.trainable_variables))

        grad_c = tape.gradient(c_loss, classifer.trainable_variables)
        c_optimizer.apply_gradients(zip(grad_c, classifer.trainable_variables))

        grad_f = tape.gradient(E_d, feature.trainable_variables)
        f_optimizer.apply_gradients(zip(grad_f, feature.trainable_variables))

        if epoch % 200 == 0:
            loss_c.append(float(c_loss))
            loss_d.append(float(d_loss))
            correct_s_c = 0
            correct_st_d = 0
            correct_t_c = 0
            total_s_c = 0
            total_st_d = 0
            total_t_c = 0

            for x in range(20):
                batch_s = next(db_s_test_iter)
                batch_st = next(db_st_test_iter)
                batch_t = next(db_t_test_iter)
                sign_test_s = batch_s[0]
                label_test_s = batch_s[1]
                sign_test_st = batch_st[0]
                domain_test_st = domain_replace(batch_st[2], batch_size, data_domain_name_s, data_domain_name_t)
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
                  'mmd_loss', float(mmd_loss),
                  'E_d loss', float(E_d),
                  's_c acc', correct_s_c / total_s_c,
                  'st_d acc', correct_st_d / total_st_d,
                  't_c acc', correct_t_c / total_t_c, )
            if epoch == epochs:
                train_data = [domain_name_s, domain_name_t, learning_rate,
                              float(mmd_loss), float(E_d), float(c_loss), float(d_loss),
                              float(correct_s_c / total_s_c), float(correct_st_d / total_st_d), correct_t_c / total_t_c]
                [format(float(i), ".4f") for i in train_data]

    feature.save_weights('./ckpt/trained/' + data_name + '/' + domain_name_s + '-' + domain_name_t + '-feature.ckpt')
    classifer.save_weights('./ckpt/trained/' + data_name + '/' + domain_name_s + '-' + domain_name_t + '-classifer.ckpt')
    print('weights save OK')

    # sys.stdout.close()

    return train_data


def test_otherdomain(test_data_domain, domain_name_s, domain_name_t):
    # sys.stdout = open('out.txt', 'a')
    # 用于迁移训练完成之后，对其他域数据的分类效果，判别模型是否提取到了共同特征
    path = r'D:/DATABASE/CWRU_make_samples/samples/crack_0.021'
    data_path = glob.glob(path + '/' + test_data_domain + '/**.mat')
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
    # sys.stdout.close()
    testdata = [test_data_domain, float(t_loss), correct_test / total]

    return testdata


def domain_replace(domain_data, batch_size, source_name, target_name):
    source_name = list(source_name.replace('-', ''))
    source_name = [int(x) for x in source_name]
    target_name = list(target_name.replace('-', ''))
    target_name = [int(x) for x in target_name]
    data_replaced = []
    for i in range(batch_size):
        if domain_data[i] in source_name:
            data_replaced.append(0)
        if domain_data[i] in target_name:
            data_replaced.append(1)
    tf.convert_to_tensor(data_replaced)

    return data_replaced


if __name__ == '__main__':

    cond = ['0', '1', '2', '3']
    cond.extend(cond)
    data = pd.DataFrame()

    for i in range(3):
        lr = 0.001
        for j in range(int(len(cond) / 2)):
            data1 = main(lr, cond[j], cond[j + 1])
            data1_1 = test_otherdomain(cond[j + 2], cond[j], cond[j + 1])
            data1.extend(data1_1)
            data1_2 = test_otherdomain(cond[j + 3], cond[j], cond[j + 1])
            data1.extend(data1_2)
            data_1 = pd.DataFrame(data1)

            data2 = main(lr, cond[j], cond[j + 2])
            data2_1 = test_otherdomain(cond[j + 1], cond[j], cond[j + 2])
            data2.extend(data2_1)
            data2_2 = test_otherdomain(cond[j + 3], cond[j], cond[j + 2])
            data2.extend(data2_2)
            data_2 = pd.DataFrame(data2)

            data3 = main(lr, cond[j], cond[j + 3])
            data3_1 = test_otherdomain(cond[j + 1], cond[j], cond[j + 3])
            data3.extend(data3_1)
            data3_2 = test_otherdomain(cond[j + 2], cond[j], cond[j + 3])
            data3.extend(data3_1)
            data_3 = pd.DataFrame(data3)
            datai = pd.concat([data_1, data_2, data_3], axis=1)

            data = pd.concat([data, datai.T], axis=0)

        app = xw.App()
        wb = app.books.add()
        sheet1 = xw.sheets["sheet1"]
        sheet1.range('A1').value = data
        wb.save('MMD_repeat'+str(i)+'.xlsx')
        wb.close()
        app.quit()

    for z in range(5):
        lr = 10**(-z)
        for j in range(int(len(cond) / 2)):
            data1 = main(lr, cond[j], cond[j + 1])
            data1_1 = test_otherdomain(cond[j + 2], cond[j], cond[j + 1])
            data1.extend(data1_1)
            data1_2 = test_otherdomain(cond[j + 3], cond[j], cond[j + 1])
            data1.extend(data1_2)
            data_1 = pd.DataFrame(data1)

            data2 = main(lr, cond[j], cond[j + 2])
            data2_1 = test_otherdomain(cond[j + 1], cond[j], cond[j + 2])
            data2.extend(data2_1)
            data2_2 = test_otherdomain(cond[j + 3], cond[j], cond[j + 2])
            data2.extend(data2_2)
            data_2 = pd.DataFrame(data2)

            data3 = main(lr, cond[j], cond[j + 3])
            data3_1 = test_otherdomain(cond[j + 1], cond[j], cond[j + 3])
            data3.extend(data3_1)
            data3_2 = test_otherdomain(cond[j + 2], cond[j], cond[j + 3])
            data3.extend(data3_1)
            data_3 = pd.DataFrame(data3)
            datai = pd.concat([data_1, data_2, data_3], axis=1)

            data = pd.concat([data, datai.T], axis=0)

        app = xw.App()
        wb = app.books.add()
        sheet1 = xw.sheets["sheet1"]
        sheet1.range('A1').value = data
        wb.save('MMD_lr'+str(lr)+'.xlsx')
        wb.close()
        app.quit()