import numpy as np
import tensorflow as tf
import glob
import matplotlib.pylab as plt
from sklearn.manifold import TSNE
from dataset import make_sign_dataset
from Mul_DACNN import Gfeature, Gy
from sklearn.metrics import confusion_matrix
import itertools
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])


def get_data(test_name, data_name, data_subname, test_data_domain, domain_name_s, domain_name_t, lamda):
    batch_size = 128
    if data_name == 'CWRU':
        path = r'D:/DATABASE/CWRU_make_samples/samples/'
    elif data_name == 'ZXJG-SC':
        path = r'D:/DATABASE/ZXJ_test_data/fault_gear_zdcs/gear_fault_standard/sample/'
    path = path + data_subname
    data_dir = glob.glob(path + '/' + test_data_domain + '/**.mat')
    """
    :return: 信号、标签
    """
    digits = make_sign_dataset(data_dir, batch_size=batch_size, dataset_name=data_name)
    data0 = iter(digits[0])
    batch = next(data0)
    sign = tf.reshape(batch[0], batch[0].shape[:2])
    label = batch[1]
    """
    载入模型，输入信号输出特征
    """
    if data_name=='CWRU':
        if test_name=='DACNN':
            test_name=''
        else:
            test_name = test_name
    feature = Gfeature()
    feature.build(input_shape=(batch_size, 4096, 1))
    feature.load_weights('D:/python_workfile/Multi-condition diagnosis/ckpt/trained/'+
                         data_name+'/'+data_subname+'/'+test_name+'-'+domain_name_s+'-'+domain_name_t+'-'+str(lamda)+
                         '-feature.ckpt')
    feature_sign = feature.call(batch[0], training=None)

    return sign, label, feature_sign


def plot_embdedding(data, label, title, normal=True):

    fig = plt.figure()
    ax = plt.subplot(111)
    if normal==True:
        x_max, x_min = np.min(data, 0), np.max(data, 0)
        data = (data-x_min) / (x_max - x_min)
    if normal==None:
        axes = [np.min(data[:, 0])-10, np.max(data[:, 0])+10, np.min(data[:, 1])-10, np.max(data[:, 1])+10]
        plt.axis(axes)
    label = label.numpy()
    # 遍历所有样本
    for i in range(data.shape[0]):
        # plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i]/10), fontsize=10)
        shape_list=['o', '^', ',', 'D', '+', 'v']
        color_list = ['b', 'g', 'r', 'c', 'y', 'm']
        plt.scatter(data[i, 0], data[i, 1], marker=shape_list[label[i]], c=color_list[label[i]])
    # plt.autoscale(True)
    plt.xticks()
    plt.yticks()
    plt.title(title, fontsize=14)

    return fig


def t_sne(test_name, data_name, data_subname, test_data_domain, domain_name_s, domain_name_t, lamda):

    sign, label, feature = get_data(test_name, data_name, data_subname, test_data_domain, domain_name_s, domain_name_t, lamda)
    ts = TSNE(n_components=2, init='pca', random_state=0)
    result = ts.fit_transform(feature)
    fig = plot_embdedding(result, label, "t_SNE Embedding of features", normal=None)
    text = 'model_name:'+test_name+'\n'\
           +'dataset_name:'+data_name+'_'+data_subname+'_'+test_data_domain+'\n'\
           +'source-target:'+domain_name_s+'-'+domain_name_t+'\n'\
           +'lambda:'+lamda
    plt.text(0.05, 0.05, text)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_num, title = "Confusion matrix",
                          cmap=plt.cm.Blues, save_flg=False):
    # 参数 y_true为测试数据集的真实标签，y_pred为网络对测试数据集的预测结果
    classes = [str(i) for i in range(class_num)]
    # 参数i的取值范围根据你自己数据集的划分类别来修改，我这儿为7代表数据集共有7类
    labels = range(class_num)
    # 数据集的标签类别，跟上面I对应
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = plt.figure(figsize=(14, 12))
    h = plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=40)
    hm = plt.colorbar(h)
    hm.ax.tick_params(labelsize=30)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=30)
    plt.yticks(tick_marks, classes, fontsize=30)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=30)
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    if save_flg:
        plt.savefig("./confusion_matrix.png")

    return fig


def confuse_matrix(test_name, data_name, data_subname, test_data_domain, domain_name_s, domain_name_t, lamda):
    if data_name == 'CWRU':
        path = r'D:/DATABASE/CWRU_make_samples/samples/'
    elif data_name == 'ZXJG-SC':
        path = r'D:/DATABASE/ZXJ_test_data/fault_gear_zdcs/gear_fault_standard/sample/'
    path = path + data_subname
    data_path = glob.glob(path + '/' + test_data_domain + '/**.mat')
    batch_size = 256

    test_data, _, _, _, test_num, _ = make_sign_dataset(data_path, batch_size, data_name)
    data = test_data.repeat()
    data = iter(data)

    if test_name=='DACNN':
        test_name=''
    else:
        test_name = test_name+'-'
    feature = Gfeature()
    feature.build(input_shape=(batch_size, 4096, 1))
    feature.load_weights('./ckpt/trained/' + data_name + '/' + data_subname + '/' + test_name
                         + domain_name_s + '-' + domain_name_t + '-' + str(lamda) + '-feature.ckpt')
    classifer = Gy()
    classifer.build(input_shape=(batch_size, 4096))
    classifer.load_weights('./ckpt/trained/' + data_name + '/' + data_subname + '/' + test_name
                           + domain_name_s + '-' + domain_name_t + '-' + str(lamda) + '-classifer.ckpt')

    batch_x = next(data)
    sign = batch_x[0]
    label_test = tf.one_hot(batch_x[1], depth=6)
    label_pred_test = classifer.call(feature.call(sign, training=None), training=None)

    label_pred = tf.argmax(label_pred_test, axis=-1)
    label_real = tf.argmax(label_test, axis=-1)
    fig = plot_confusion_matrix(label_real, label_pred, class_num=6)
    text = 'model_name:' + test_name + '\n' \
           + 'dataset_name:' + data_name + '_' + data_subname + '_' + test_data_domain + '\n' \
           + 'source-target:' + domain_name_s + '-' + domain_name_t + '\n' \
           + 'lambda:' + lamda
    plt.text(0, 0, text)
    plt.show()



if __name__ == '__main__':
    test_name = 'LWD'
    data_name = 'CWRU'
    data_subname = 'crack_0.007'
    test_domain = '0-1'
    source_domian = '0'
    target_domain = '1'
    lamda = '1'
    t_sne(test_name, data_name, data_subname, test_domain, source_domian, target_domain, lamda)
    confuse_matrix(test_name, data_name, data_subname, test_domain, source_domian, target_domain, lamda)
