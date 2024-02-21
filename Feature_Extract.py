import tensorflow as tf
from Mul_DACNN import Gfeature, Gy
import glob
from dataset import make_sign_dataset
import matplotlib.pyplot as plt
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])

def feature_extractor(test_name, data_name, data_subname, test_data_domain, domain_name_s, domain_name_t, lamda):
    # 用于迁移训练完成之后，对其他域数据的分类效果，判别模型是否提取到了共同特征
    if data_name == 'CWRU':
        path = r'D:/DATABASE/CWRU_xjs/sample/12k/Drive_End/noise/snr5'
    elif data_name == 'ZXJG-SC':
        path = r'D:/DATABASE/ZXJ_test_data/fault_gear_zdcs/gear_fault_standard/sample/'
    data_path = glob.glob(path + '/' + data_subname + '/' + test_data_domain + '/**.mat')
    batch_size = 64
    print('data_name', data_name, 'data_subname', data_subname,
          'load condition', test_data_domain, 'samples number', len(data_path))

    test_data, _, _, _, test_num, _ = make_sign_dataset(data_path, batch_size, data_name)
    data = test_data.repeat()
    data = iter(data)

    feature = Gfeature()
    feature.build(input_shape=(batch_size, 4096, 1))
    feature.load_weights('./ckpt/trained3_cnn1&cnn2/' + data_name + '/' + data_subname + '/' + test_name + '-'
                         + domain_name_s + '-' + domain_name_t + '-' + str(lamda) + '-feature.ckpt')
    classifer = Gy()
    classifer.build(input_shape=(batch_size, 4096))
    classifer.load_weights('./ckpt/trained3_cnn1&cnn2/' + data_name + '/' + data_subname + '/' + test_name + '-'
                           + domain_name_s + '-' + domain_name_t + '-' + str(lamda) + '-classifer.ckpt')
    print('model source domain', domain_name_s, 'target domain', domain_name_t)


    batch_x = next(data)
    sign = batch_x[0]
    label_test = batch_x[1]
    feature = feature.call(sign, training=None)

    feature_data = feature.numpy()
    label_data = label_test.numpy()

    return feature_data, label_test


def main():
    test_name = 'cnn1_pretrain_noise_snr5'
    data_name = 'CWRU'
    data_subname = '007'
    test_data_domain = '2'
    domain_s = '1'
    domain_t = '2'
    lamda = 0.01
    feature, label = feature_extractor(test_name, data_name, data_subname, test_data_domain, domain_s, domain_t, lamda)
    class_num = 6
    repeat_num = 3
    data = {}
    plt.figure(1, figsize=(16, 6))
    for i in range(class_num):
        label_list = np.where(label == i)
        data[str(i)] = feature[label_list[0][0:repeat_num]]
        for j in range(0, repeat_num):
            plt.subplot(repeat_num, class_num, i+1+class_num*j)
            # 颜色有默认、tomato、seagreen
            plt.plot(data[str(i)][j], color='seagreen')
            plt.yticks([0, 20])
    plt.show()


if __name__ == '__main__':
    main()