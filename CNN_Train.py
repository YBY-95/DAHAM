import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
import glob
from Mul_DACNN import Gfeature, Gy, Gfeature1, Gfeature2
from DCNN import DCNN
from dataset import make_sign_dataset
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

def main(model_name, train_name, datasetname, dataset_subname, data_name):
    # data_name 指的是域名称；datasetname 指的是数据集的整体大名称（如CWRU等）；dataset_subname指的是大数据集分割为小数据集后的名称。

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    epochs = 15000  # 训练次数上限
    batch_size = 128  # batch size
    learning_rate = 0.001
    is_training = True
    class_num = 8
    print('model name :', model_name,
          '\ntrain name', train_name,
          '\ndataset name：', datasetname,
          '\ndata_subname:', dataset_subname,
          '\ndomain_name', data_name)
    # 获取数据集路径
    if datasetname == 'CWRU':
        path = r'D:/DATABASE/CWRU_xjs/sample/12k/Drive_End/'
    elif datasetname == 'ZXJG-SC':
        path = r'D:/DATABASE/ZXJ_test_data/fault_gear_zdcs/gear_fault_standard/sample/'
    elif datasetname == 'ZXJB':
        path = r'D:/DATABASE/ZXJ_test_data/fault_bearing_standard_sample/sample'
    signal_path = glob.glob(os.path.join(path, dataset_subname, data_name)+r'/**.mat')
    print('number of signal samples=', len(signal_path))

    # 构建源域数据集对象
    train_set, test_set, len_dataset, whole_data, train_num, test_num = make_sign_dataset(signal_path, batch_size,
                                                                                          dataset_name=datasetname)
    print(len_dataset)
    dataset = train_set.repeat()  # 重复循环
    db_iter = iter(dataset)
    dataset_t = test_set.repeat()  # 重复循环
    db_t_iter = iter(dataset_t)

    if train_name == 'DCNN':
        # 将DCNN作为整体训练
        feature = DCNN()
        feature.build(input_shape=(batch_size, 4096, 1))
        CNN_optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

        for epoch in range(epochs+1):  # 训练epochs次
            # 2. 训练分类器
            # 采样源域信号
            batch_x = next(db_iter)  # 采样源域信号
            sign = batch_x[0]
            label = tf.one_hot(batch_x[1], depth=class_num)
            # 分类器前向计算
            loss = []
            with tf.GradientTape(persistent=True) as tape:
                label_pred = feature.call(sign, is_training)
                crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
                y_loss = crossentro(label, label_pred)
            grad = tape.gradient(y_loss, feature.trainable_variables)
            CNN_optimizer.apply_gradients(zip(grad, feature.trainable_variables))

            correct = tf.cast(
                tf.equal(
                    tf.argmax(label_pred, axis=-1),
                    tf.argmax(label, axis=-1)), tf.float32)
            correct_train = np.sum(correct)

            # 可视化 损失 准确率
            if epoch % 500 == 0:
                loss.append(float(y_loss))
                correct_test = 0
                total = 0

                for x in range(test_num // batch_size):
                    batch_x = next(db_t_iter)
                    sign_test = batch_x[0]
                    label_test = tf.one_hot(batch_x[1], depth=class_num)
                    label_pred_test = feature.call(sign_test, training=None)
                    correct = tf.cast(
                        tf.equal(
                            tf.argmax(label_pred_test, axis=-1),
                            tf.argmax(label_test, axis=-1)), tf.float32)
                    correct_test += np.sum(correct)
                    total += batch_size
                    t_loss = tf.reduce_mean(crossentro(label_test, label_pred_test))

                print(epoch,
                      'y_loss:', float(y_loss),
                      'test_loss', float(t_loss),
                      'test_acc', correct_test / total,
                      'train_acc', correct_train / batch_size)
            # if correct_test / total > 0.85:
            #     gfeature.save_weights('gfeature.ckpt')
            #     yclassifer.save_weights('yclassifer.ckpt')
            #     sys.exit(0)

    elif train_name == 'CNN_pretrain':
        # domain adversarial 的预训练，特征提取器与分类器分开训练

        # 创建特征提取网络
        if model_name == 'DCNN':
            gfeature = Gfeature()
            gfeature.build(input_shape=(batch_size, 4096, 1))
            yclassifer = Gy()
            yclassifer.build(input_shape=(batch_size, 4096))
        elif model_name == 'cnn1':
            gfeature = Gfeature1()
            gfeature.build(input_shape=(batch_size, 4096, 1))
            yclassifer = Gy()
            yclassifer.build(input_shape=(batch_size, 4096))
        elif model_name == 'cnn2':
            gfeature = Gfeature2()
            gfeature.build(input_shape=(batch_size, 4096, 1))
            yclassifer = Gy()
            yclassifer.build(input_shape=(batch_size, 4096))

        # 创建优化器
        f_optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
        y_optimizer = keras.optimizers.SGD(learning_rate=learning_rate)

        for epoch in range(epochs+1):  # 训练epochs次
            # 2. 训练分类器
            # 采样源域信号
            batch_x = next(db_iter)  # 采样信号
            sign = batch_x[0]
            label = tf.one_hot(batch_x[1], depth=class_num)
            # 分类器前向计算
            loss = []
            with tf.GradientTape(persistent=True) as tape:
                feature = gfeature.call(sign, is_training)
                label_pred = yclassifer.call(feature, is_training)
                crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
                y_loss = crossentro(label, label_pred)
            grad_f = tape.gradient(y_loss, gfeature.trainable_variables)
            grad_y = tape.gradient(y_loss, yclassifer.trainable_variables)
            f_optimizer.apply_gradients(zip(grad_f, gfeature.trainable_variables))
            y_optimizer.apply_gradients(zip(grad_y, yclassifer.trainable_variables))

            correct = tf.cast(
                tf.equal(
                    tf.argmax(label_pred, axis=-1),
                    tf.argmax(label, axis=-1)), tf.float32)
            correct_train = np.sum(correct)

            # 可视化 损失 准确率
            if epoch % 1000 == 0:
                loss.append(float(y_loss))
                correct_test = 0
                total = 0

                for x in range(test_num // batch_size):
                    batch_x = next(db_t_iter)
                    sign_test = batch_x[0]
                    label_test = tf.one_hot(batch_x[1], depth=class_num)
                    label_pred_test = yclassifer.call(gfeature.call(sign_test, training=None), training=None)
                    correct = tf.cast(
                        tf.equal(
                            tf.argmax(label_pred_test, axis=-1),
                            tf.argmax(label_test, axis=-1)), tf.float32)
                    correct_test += np.sum(correct)
                    total += batch_size
                    t_loss = tf.reduce_mean(crossentro(label_test, label_pred_test))

                print(epoch,
                      'y_loss:', float(y_loss),
                      'test_loss', float(t_loss),
                      'test_acc', correct_test / total,
                      'train_acc', correct_train / batch_size)


    if os.path.exists(r'./ckpt/pretrain/' + datasetname + model_name + '/' + dataset_subname) is None:
        os.mkdir(r'./ckpt/pretrain/' + datasetname + model_name + '/' + dataset_subname)
    gfeature.save_weights(r'./ckpt/pretrain/' + datasetname+ model_name + '/' + dataset_subname + '/'+data_name+'-feature.ckpt')
    yclassifer.save_weights(r'./ckpt/pretrain/' + datasetname+ model_name + '/' + dataset_subname + '/'+data_name+'-classifer.ckpt')
    print('CNN weights saved')

def pretrain_test(model_name, datasetname, dataset_subname, pretrain_data_name, test_data_name):
    batch_size = 64
    if datasetname == 'CWRU':
        signal_path = glob.glob(r'D:/DATABASE/CWRU_xjs/sample/12k/Drive_End/' + dataset_subname+r'/'+test_data_name+r'/**.mat')
    elif datasetname == 'ZXJG-SC':
        signal_path = glob.glob(r'D:/DATABASE/ZXJ_test_data/fault_gear_zdcs/gear_fault_standard/sample/'
                                +dataset_subname+'/'+test_data_name+r'/**.mat')
    elif datasetname == 'ZXJB':
        signal_path = glob.glob(r'D:/DATABASE/ZXJ_test_data/fault_bearing_standard_sample/sample/'
                                +dataset_subname+'/'+test_data_name+r'/**.mat')

    if model_name == 'DCNN':
        feature = Gfeature()
        feature.build(input_shape=(batch_size, 4096, 1))
        classifer = Gy()
        classifer.build(input_shape=(batch_size, 4096))
    elif model_name == 'cnn1':
        feature = Gfeature1()
        feature.build(input_shape=(batch_size, 4096, 1))
        classifer = Gy()
        classifer.build(input_shape=(batch_size, 4096))
    elif model_name == 'cnn2':
        feature = Gfeature2()
        feature.build(input_shape=(batch_size, 4096, 1))
        classifer = Gy()
        classifer.build(input_shape=(batch_size, 4096))
    feature.load_weights(r'./ckpt/pretrain/' + datasetname + model_name + '/' + dataset_subname + '/'+pretrain_data_name+'-feature.ckpt')
    classifer.load_weights(r'./ckpt/pretrain/' + datasetname + model_name + '/' + dataset_subname + '/'+pretrain_data_name+'-classifer.ckpt')
    train_set, test_set, len_dataset, whole_data, train_num, test_num = make_sign_dataset(signal_path,
                                                                                          batch_size,
                                                                                          datasetname)
    dataset = train_set.repeat()
    db_iter = iter(dataset)

    correct_test = 0
    total = 0
    crossentro = tf.keras.losses.CategoricalCrossentropy()

    for x in range(train_num // batch_size):
        batch_x = next(db_iter)
        sign = batch_x[0]
        label_test = tf.one_hot(batch_x[1], depth=8)
        label_pred_test = classifer.call(feature.call(sign, training=None), training=None)
        correct = tf.cast(
            tf.equal(
                tf.argmax(label_pred_test, axis=-1),
                tf.argmax(label_test, axis=-1)), tf.float32)
        correct_test += np.sum(correct)
        total += batch_size
        t_loss = tf.reduce_mean(crossentro(label_test, label_pred_test))

    print('train data:', pretrain_data_name, 'test data:', test_data_name,
          'loss', float(t_loss),
          'acc', correct_test / total)


if __name__ == '__main__':
    model_name = 'cnn1'
    train_name = 'CNN_pretrain'
    # dataname = 'CWRU'
    # domian_name = ['0', '1', '2', '3']
    # data_subname = ['007']

    # dataname = 'ZXJG-SC'
    # domian_name = ['_1000_0_', '_1000_15_', '_1000_30_',
    #             '_1500_0_', '_1500_15_', '_1500_30_',
    #             '_2000_0_', '_2000_15_', '_2000_30_']

    dataname = 'ZXJB'
    domain_name = ['_1000_0_', '_1000_20_', '_1000_40_',
                   '_1500_0_', '_1500_20_', '_1500_40_',
                   '_2000_0_', '_2000_20_', '_2000_40_',
                   '_2500_0_', '_2500_20_', '_2500_40_',
                   '_3000_0_', '_3000_20_', '_3000_40_']
    data_subname = ['without_box']
    for i in data_subname:
        for j in domain_name:
            # if j in []:
            #     continue
            main(model_name, train_name, dataname, i, j)
            for k in domain_name:
                if k != i:
                    pretrain_test(model_name, dataname, i, j, k)

