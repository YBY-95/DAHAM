import tensorflow as tf
from tensorflow import keras
from Mul_DACNN import Gfeature, Gfeature1, Gfeature2, Gy, GdWD
import os
import glob
from dataset import make_sign_dataset
import numpy as np
import xlwings as xw
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pylab as plt
from t_SNE_plot import plot_embdedding, plot_confusion_matrix
import datetime
from sklearn.metrics import confusion_matrix
import itertools
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])

def gradient_penalty(discriminator, batch_s, batch_t):

    batchsz = batch_s.shape[0]

    # [b , w, c]
    t = tf.random.uniform([batchsz, 1, 1], dtype=tf.dtypes.float64)
    # [b, 1, 1] => [b, w, c]
    t = tf.broadcast_to(t, batch_s.shape)

    interpalte = t * batch_s + (1 - t) * batch_t
    interpalte = interpalte[:, :, 0]

    with tf.GradientTape() as tape:
        tape.watch([interpalte])
        d_interplate_logits = discriminator(interpalte)
    grads = tape.gradient(d_interplate_logits, interpalte)

    # [b, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1)
    gp = tf.reduce_mean((gp-1)**2)

    return gp


def local_sample_divide(signal_batch, class_num):
    signal = signal_batch[0]
    lable = signal_batch[1]
    divide_signal = []
    sample_batch = []
    index_len = list()
    for i in range(class_num):
        index = tf.where(tf.equal(lable, i))
        signal_sameclass = tf.gather(signal, axis=0, indices=index)
        divide_signal.append(signal_sameclass[:, 0, :, 0])
        index_len.append(len(index))
    for x in divide_signal:
        if len(x) == 0:
            sample_batch.append(tf.cast(tf.zeros([len(signal), 4096]), tf.float64))
            continue
        dup_times = len(signal)//len(x) + 1
        y = tf.tile(x, [dup_times, 1])
        sample_batch.append(y[:len(signal), :])

    # for m in range(min(index_len)):
    #     if len(signal) % (min(index_len)-m) == 0:
    #         divide_len = min(index_len) - m
    #         dup_times = int(len(signal)/divide_len)
    #         for j in divide_signal:
    #             sample_batch.append(tf.tile(j[:divide_len, :], [dup_times, 1]))
    #         break
    #
    #     if min(index_len) == 0:
    #         sample_batch.append(tf.zeros([len(signal), class_num]))
    #         break

    sign_batch = tf.random.shuffle(tf.reshape(sample_batch, [len(sample_batch), len(sample_batch[1]), -1]))

    return sign_batch


def local_w_distance(feature, classifer, discriminator, batch_s, batch_t, class_num, batch_size):
    sign_t = batch_t[0]
    wd_weight = classifer(feature(sign_t))
    sign_s = local_sample_divide(batch_s, class_num)
    wd_metric = np.empty([class_num, batch_size])
    wd_t = discriminator(feature(tf.reshape(sign_t, [batch_size, 4096, 1])))
    for i in range(class_num):
        wd_s = discriminator(feature(tf.reshape(sign_s[i], [batch_size, 4096, 1])))
        wd = wd_s - wd_t
        wd_metric[i] = wd.numpy().T
    wd_metric = tf.cast(tf.convert_to_tensor(wd_metric), tf.float32)
    local_w_loss = tf.reduce_mean(tf.reduce_sum(wd_weight*tf.transpose(wd_metric), 1))

    return local_w_loss


def gradient_reversal_layer(gradient_list):
    for i in range(len(gradient_list)):
        if i == 0:
            b = []
        a = -1*np.array(gradient_list[i])
        b.append(tf.convert_to_tensor(a))
    return b


def main(test_name, dataset_name, dataset_subname, domain_name_s, domain_name_t, lamda, omega,
         tensorboard_log_dir='./tensorboard/'):

    # 输出写入out.txt文件中,同时run界面中的输出关闭
    # sys.stdout = open('out.txt', 'a')

    # 设置参数
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    epochs = 10000  # 训练步数上限
    batch_size = 256  # batch size
    class_num = 8
    learning_rate = 0.001
    is_training = True
    mu = 10
    tensorboard = True
    print('crack_size', dataset_subname)
    if dataset_name == 'CWRU':
        path = r'D:/DATABASE/CWRU_xjs/sample/12k/Drive_End/'
        signal_path = path + dataset_subname
        domain_name_st = domain_name_s + '-' + domain_name_t
    elif dataset_name == 'ZXJG-SC':
        path = r'D:/DATABASE/ZXJ_test_data/fault_gear_zdcs/gear_fault_standard/sample/'
        signal_path = path + dataset_subname
        domain_name_st = domain_name_s + '--' + domain_name_t
    elif dataset_name == 'ZXJB':
        path = r'D:/DATABASE/ZXJ_test_data/fault_bearing_standard_sample/sample/'
        signal_path = path + dataset_subname
        domain_name_st = domain_name_s + '-' + domain_name_t

    print('lamda = ', lamda)
    print('source domain:', domain_name_s, '  target domain:', domain_name_t)

    # 获取数据集路径
    signal_s_path = glob.glob(signal_path + r'/' + domain_name_s + r'/**.mat')
    print('source domain signals num:', len(signal_s_path))
    signal_st_path = glob.glob(signal_path + r'/' + domain_name_st + r'/**.mat')
    print('source&target domain signals num:', len(signal_st_path))
    signal_t_path = glob.glob(signal_path + r'/' + domain_name_t + r'/**.mat')
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
    if 'cnn1' in test_name:
        feature = Gfeature1()
        feature.build(input_shape=(batch_size, 4096, 1))
        classifer = Gy()
        classifer.build(input_shape=(batch_size, 4096))
        feature.load_weights(r'D:/python_workfile/Multi-condition diagnosis/ckpt/pretrain/'
                             + dataset_name + 'cnn1' + '/' + dataset_subname + '/' + domain_name_s + '-feature.ckpt')
        classifer.load_weights(r'D:/python_workfile/Multi-condition diagnosis/ckpt/pretrain/'
                               + dataset_name + 'cnn1' + '/' + dataset_subname + '/' + domain_name_s + '-classifer.ckpt')
        print('load weights OK')
        discrim = GdWD()
        discrim.build(input_shape=(batch_size, 4096))
        # 创建域分类器
    elif 'cnn2' in test_name:
        feature = Gfeature2()
        feature.build(input_shape=(batch_size, 4096, 1))
        classifer = Gy()
        classifer.build(input_shape=(batch_size, 4096))
        feature.load_weights(r'D:\python_workfile\Multi-condition diagnosis\ckpt\pretrain'
                             + 'CWRUcnn2' + '/' + dataset_subname + '/' + domain_name_s + '-feature.ckpt')
        classifer.load_weights(
            './ckpt/pretrain/' + 'CWRUcnn2' + '/' + dataset_subname + '/' + domain_name_s + '-classifer.ckpt')
        print('load weights OK')
        discrim = GdWD()
        discrim.build(input_shape=(batch_size, 4096))
    else:
        feature = Gfeature()
        feature.build(input_shape=(batch_size, 4096, 1))
        classifer = Gy()
        classifer.build(input_shape=(batch_size, 4096))
        feature.load_weights(
            './ckpt/pretrain/' + dataset_name + '3nd' + '/' + dataset_subname + '/' + domain_name_s + '-feature.ckpt')
        classifer.load_weights(
            './ckpt/pretrain/' + dataset_name + '3nd' + '/' + dataset_subname + '/' + domain_name_s + '-classifer.ckpt')
        print('load weights OK')
        discrim = GdWD()
        discrim.build(input_shape=(batch_size, 4096))
    print('model building OK')

    # 创建优化器
    f_optimizer = keras.optimizers.RMSprop(learning_rate=0.1 * learning_rate)
    c_optimizer = keras.optimizers.SGD(learning_rate=0.1 * learning_rate)
    d_optimizer = keras.optimizers.RMSprop(learning_rate=learning_rate)

    loss_wd = []
    loss_wdlocal = []
    loss_c = []

    if tensorboard:
        train_log_dir = tensorboard_log_dir + domain_name_st + '-' + str(lamda) + '/train'
        test_log_dir = tensorboard_log_dir + domain_name_st + '-' + str(lamda) + '/test'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir, name=domain_name_st+'-'+str(lamda))
        test_summary_writer = tf.summary.create_file_writer(test_log_dir, name=domain_name_st+'-'+str(lamda))

    for epoch in range(epochs + 1):
        batch_s = next(db_s_train_iter)
        batch_st = next(db_st_train_iter)
        batch_t = next(db_t_train_iter)

        sign_s = batch_s[0]
        sign_t = batch_t[0]

        label_real = tf.one_hot(batch_s[1], depth=class_num)
        if dataset_name == 'CWRU':
            domain_test_st = domain_replace(batch_st[2], batch_size, domain_name_s, domain_name_t)
        else:
            domain_test_st = batch_st[1]
        Cate_crossentro = keras.losses.CategoricalCrossentropy(from_logits=None)
        with tf.GradientTape(persistent=True) as tape:
            feature_s = feature.call(sign_s, is_training)
            label_pred = classifer.call(feature_s, is_training)
            feature_t = feature.call(sign_t, is_training)
            wd_domain_pred_t = discrim.call(feature_t, is_training)
            wd_domain_pred_s = discrim.call(feature_s, is_training)

            c_loss = Cate_crossentro(label_real, label_pred)
            # c_loss 源域分类损失
            gp = tf.cast(gradient_penalty(discrim, sign_s, sign_t), tf.float32)
            wd_loss = tf.reduce_mean(wd_domain_pred_s)-tf.reduce_mean(wd_domain_pred_t)
            # wd_loss W距离全局域判别损失
            wd_localloss = local_w_distance(feature, classifer, discrim, batch_s, batch_t, class_num, batch_size)
            # wd_localloss W距离局部域判别损失

            d_loss = omega*wd_loss + (1-omega)*wd_localloss - mu*gp
            E_d = c_loss + lamda*d_loss

        grad_d = gradient_reversal_layer(tape.gradient(d_loss, discrim.trainable_variables))
        d_optimizer.apply_gradients(zip(grad_d, discrim.trainable_variables))

        if epoch%1==0:
            grad_c = tape.gradient(c_loss, classifer.trainable_variables)
            c_optimizer.apply_gradients(zip(grad_c, classifer.trainable_variables))

            grad_f = tape.gradient(E_d, feature.trainable_variables)
            f_optimizer.apply_gradients(zip(grad_f, feature.trainable_variables))

        if tensorboard:
            with train_summary_writer.as_default():
                tf.summary.scalar('c_loss_train', float(c_loss), step=epoch)
                tf.summary.scalar('wd_loss_train', float(wd_loss), step=epoch)
                tf.summary.scalar('local_wd_loss_train', float(wd_localloss), step=epoch)
                tf.summary.scalar('gp_train', float(gp), step=epoch)
                tf.summary.scalar('d_loss_train', float(d_loss), step=epoch)
                tf.summary.scalar('f_loss_train', float(E_d), step=epoch)

        if epoch % 1000 == 0:
            loss_c.append(float(c_loss))
            loss_wd.append(float(wd_loss))
            loss_wdlocal.append((float(wd_localloss)))
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
                  'wd-loss:', float(wd_loss),
                  'gp', float(gp),
                  'wd_localloss', float(wd_localloss),
                  'omega', float(omega),
                  's_c acc', correct_s_c / total_s_c,
                  'st_d acc', correct_st_d / total_st_d,
                  't_c acc', correct_t_c / total_t_c, )

            if tensorboard:
                with test_summary_writer.as_default():
                    tf.summary.scalar('sc_acc', float(correct_s_c / total_s_c), step=epoch)
                    tf.summary.scalar('st_d acc', float(correct_st_d / total_st_d), step=epoch)
                    tf.summary.scalar('t_c acc', float(correct_t_c / total_t_c), step=epoch)

            if epoch == epochs:
                train_data = [domain_name_s, domain_name_t, str(lamda),float(omega), float(c_loss), float(wd_loss), float(wd_localloss),
                              float(correct_s_c / total_s_c), float(correct_st_d / total_st_d),
                              correct_t_c / total_t_c]
                # [format(float(i), ".4f") for i in train_data]

            # t_sne降维
            # dir_fig = './picture'+'/'+test_name+'/'+dataset_subname+'/'
            # if os.path.exists(dir_fig) is False:
            #     os.makedirs(dir_fig)
            # sign = batch_st[0]
            # label = batch_st[1]
            # domain = batch_st[2]
            # feature_st = feature.call(sign, training=None)
            # ts = TSNE(n_components=2, init='pca', random_state=0)
            # result = ts.fit_transform(feature_st)
            # fig = plot_embdedding(result, label, "t_SNE Embedding of features", normal=True)
            # text = 'model_name:' + test_name + '\n' \
            #        + 'dataset_name:' + dataset_name + '_' + dataset_subname + '_' + domain_name_st + '\n' \
            #        + 'source-target:' + domain_name_s + '-' + domain_name_t + '\n' \
            #        + 'lambda:' + str(lamda) + '\n' + \
            #        'epoch:' + str(epoch)
            # plt.text(1.2, 0, text)
            # plt.savefig(dir_fig + domain_name_st + 't-sne' + str(epoch))

            # 混淆矩阵
            # batch_t = next(db_t_test_iter)
            # sign_test_t = batch_t[0]
            # labelr = batch_t[1]
            # labelp = tf.argmax(classifer.call(feature.call(sign_test_t, training=None), training=None), axis=-1)
            # fig2 = plot_confusion_matrix(labelr, labelp, class_num=class_num)
            # # text = 'model_name:' + test_name + '\n' \
            # #        + 'dataset_name:' + dataset_name + '_' + dataset_subname + '_' + domain_name_st + '\n' \
            # #        + 'source-target:' + domain_name_s + '-' + domain_name_t + '\n' \
            # #        + 'lambda:' + str(lamda) + '\n' +\
            # #        'epoch:' + str(epoch)
            # # plt.text(1.2, 0, text)
            # plt.savefig(dir_fig + domain_name_st + 'condusion_matrix' + str(epoch))

    ckpt_dir = './ckpt/trained4/' + dataset_name + '/' + dataset_subname
    if os.path.exists(ckpt_dir) is None:
        os.mkdir(ckpt_dir)
    feature.save_weights(ckpt_dir + '/' +
                         test_name + '-' + domain_name_s + '-' + domain_name_t + '-' + str(lamda)  + '-feature.ckpt')
    classifer.save_weights(ckpt_dir + '/' +
                           test_name + '-' + domain_name_s + '-' + domain_name_t + '-' + str(lamda) + '-classifer.ckpt')
    print('weights save OK')

    # sys.stdout.close()

    return train_data


def test_otherdomain(test_name, data_name, data_subname, test_data_domain, domain_name_s, domain_name_t, lamda, omega):
    # sys.stdout = open('out.txt', 'a')
    # 用于迁移训练完成之后，对其他域数据的分类效果，判别模型是否提取到了共同特征
    if data_name == 'CWRU':
        path = r'D:/DATABASE/CWRU_xjs/sample/12k/Drive_End/'
    elif data_name == 'ZXJG-SC':
        path = r'D:/DATABASE/ZXJ_test_data/fault_gear_zdcs/gear_fault_standard/sample/'
    elif data_name == 'ZXJB':
        path = r'D:/DATABASE/ZXJ_test_data/fault_bearing_standard_sample/sample/'

    data_path = glob.glob(path + '/' + data_subname + '/' + test_data_domain + '/**.mat')
    batch_size = 32
    print('data_name', data_name, 'data_subname', data_subname,
          'load condition', test_data_domain, 'samples number', len(data_path))

    test_data, _, _, _, test_num, _ = make_sign_dataset(data_path, batch_size, data_name)
    data = test_data.repeat()
    data = iter(data)

    feature = Gfeature()
    feature.build(input_shape=(batch_size, 4096, 1))
    feature.load_weights('./ckpt/trained4/' + data_name + '/' + data_subname + '/' + test_name + '-'
                         + domain_name_s + '-' + domain_name_t + '-' + str(lamda) + '-feature.ckpt')
    classifer = Gy()
    classifer.build(input_shape=(batch_size, 4096))
    classifer.load_weights('./ckpt/trained4/' + data_name + '/' + data_subname + '/' + test_name + '-'
                           + domain_name_s + '-' + domain_name_t + '-' + str(lamda) + '-classifer.ckpt')
    print('model source domain', domain_name_s, 'target domain', domain_name_t)
    correct_test = 0
    total = 0
    crossentro = tf.keras.losses.CategoricalCrossentropy()

    for x in range(test_num // batch_size):
        batch_x = next(data)
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

    test_name = 'LWD_cnn1'
    data_name = 'ZXJB'
    data_subname = 'without_box'
    if data_name=='ZXJG-SC':
        cond = ['_1000_0_', '_1000_15_', '_1000_30_',
                '_1500_0_', '_1500_15_', '_1500_30_',
                '_2000_0_', '_2000_15_', '_2000_30_']
    elif data_name=='ZXJB':
        cond = ['_1000_0_', '_1000_20_', '_1000_40_',
                '_1500_0_', '_1500_20_', '_1500_40_',
                '_2000_0_', '_2000_20_', '_2000_40_',
                '_2500_0_', '_2500_20_', '_2500_40_',
                '_3000_0_', '_3000_20_', '_3000_40_']
        # cond = ['_1000_0_', '_1000_20_',
        #         '_1500_0_', '_1500_20_']
    else:
        cond = ['0', '1', '2', '3']
    lamda_list = [0.01]
    omega_list = [0.5]
    data = pd.DataFrame()

    tensorboard_log_dir = 'tensorboard/' + test_name + '_' + data_name + '_' + data_subname + '/' \
                          + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    for domain_s in cond:
        for domain_t in cond:
            # 跳过相同域、及转速不同域/载荷不同域
            if domain_s == domain_t:
                continue
            if domain_s.split('_')[1] != domain_t.split('_')[1]:
                continue
            data1 = main(test_name, data_name, data_subname, domain_s, domain_t,
                         lamda_list[0], omega_list[0],
                         tensorboard_log_dir=tensorboard_log_dir)
            for domain_test in cond:  # 测试域
                if domain_test == domain_t:
                    continue
                if domain_test.split('_')[1] != domain_t.split('_')[1]:
                    continue
                data1_1 = test_otherdomain(test_name, data_name, data_subname,
                                           domain_test, domain_s, domain_t, lamda_list[0], omega_list[0])
                data1.extend(data1_1)
            data_1 = pd.DataFrame(data1)
            if cond.index(domain_s) == 0:
                datai = pd.DataFrame()
                datai = pd.concat([datai, data_1], axis=1)
            else:
                datai = pd.concat([datai, data_1], axis=1)
        data = pd.concat([data, datai.T], axis=0)

    app = xw.App()
    wb = app.books.add()
    sheet1 = xw.sheets["sheet1"]
    sheet1.range('A1').value = data
    wb.save(test_name + '-' + data_subname + '-' + data_name +
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx')
    wb.close()
    app.quit()


    # # 迭代迁移
    # for l in data_subname:
    #     tensorboard_log_dir = 'tensorboard/' + test_name + '_' + data_name + '_' + l + '/' \
    #                           + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
    #     for i in lamda_list:
    #         for n in omega_list:
    #             for j in range(len(cond)): # 源域
    #                 for k in range(len(cond)):  # 目标域
    #                     if j == k:
    #                         continue
    #                     else:
    #                         data1 = main(test_name, data_name, l, cond[j], cond[k], i, n, tensorboard_log_dir=tensorboard_log_dir)
    #                         for m in range(len(cond)):  # 测试域
    #                             data1_1 = test_otherdomain(test_name, data_name, l, cond[m], cond[j], cond[k], i, n)
    #                             data1.extend(data1_1)
    #                         data_1 = pd.DataFrame(data1)
    #                         if k==0:
    #                             datai = pd.DataFrame()
    #                             datai = pd.concat([datai, data_1], axis=1)
    #                         else:
    #                             datai = pd.concat([datai, data_1], axis=1)
    #                 data = pd.concat([data, datai.T], axis=0)
    #
    #     app = xw.App()
    #     wb = app.books.add()
    #     sheet1 = xw.sheets["sheet1"]
    #     sheet1.range('A1').value = data
    #     wb.save(test_name + '-' + l + '-' + data_name +
    #             datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.xlsx')
    #     wb.close()
    #     app.quit()