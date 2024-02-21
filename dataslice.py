import os
import pandas as pd
import numpy as np
import scipy.io as sc
import shutil
from itertools import product, zip_longest


def dataslice(data_name, file_dirc, save_dirc, trainset_rate):

    H_num = 0
    I_num = 0
    O_num = 0

    gap = 256
    sample_num = 4096

    if data_name == 'CWRU':
        sample_rate_list = ['12k', '48k']
        fault_side_list = ['Drive_End', 'Fan_End']
        fault_size_list = ['007', '014', '021', '028']
        load_list = ['_0_', '_1_', '_2_', '_3_']
        fault_list = ['IR', 'OR', 'B', 'normal']
        load_side_list = ['@3', '@6', '@12']

        for filepath, dirnames, filenames in os.walk(file_dirc):

            for filename in filenames:
                SourceData = sc.loadmat(os.path.join(filepath, filename))
                file_name = os.path.splitext(filename)
                file_num = file_name[0][int(file_name[0].rfind('_'))+1:]
                for k in list(SourceData.keys()):
                    if 'DE' in k:
                        key = k
                        break
                db = SourceData[key]
                train_db = db[:int(np.shape(db)[0]*trainset_rate)]
                test_db = db[int(np.shape(db)[0]*trainset_rate):]
                for i, j, k, l, m, n in product(sample_rate_list, fault_side_list, fault_size_list, load_list, fault_list, load_side_list):
                    if i in filename: sample_rate = i
                    if j in filename: fault_side = j
                    if k in filename: fault_size = k
                    if l in filename: load = l.replace('_', '')
                    if m in filename: fault = m
                    if n in filename: load_side = n
                if fault == 'OR':
                    if load_side == '@3':
                        sample_name = 'orth'
                    elif load_side == '@6':
                        sample_name = 'cent'
                    elif load_side == '@12':
                        sample_name = 'oppo'
                    else:
                        sample_name = ''
                        print('wrong:dont have ball fault load side')
                elif fault == 'IR':
                    sample_name = 'inne'
                elif fault == 'B':
                    sample_name = 'ball'
                elif fault == 'normal':
                    sample_name = 'norm'
                else:
                    sample_name = ''
                    print('wrong:dont have fault class info')
                sample_name = sample_name+'_'+load+'_'
                if not os.path.exists(save_dirc+'/'+sample_rate+'/'+fault_side+'/'+fault_size+'/'+load):
                    os.makedirs(save_dirc+'/'+sample_rate+'/'+fault_side+'/'+fault_size+'/'+load)
                for j in [train_db, test_db]:
                    n = np.shape(j)[0]
                    if j.shape[0] == test_db.shape[0]:
                        sample_name = 'test_'+sample_name
                    for i in range(n // gap - sample_num // gap - 1):
                        if i == 0:
                            sample_data = (j[0:sample_num]-min(j)) / (max(j)-min(j))
                        else:
                            sample_data = (j[i * gap:i * gap + sample_num]-min(j)) / (max(j)-min(j))
                        sc.savemat(os.path.join
                                   (save_dirc+'/'+sample_rate+'/'+fault_side+'/'+fault_size+'/'+load,
                                    sample_name + str(i)) + '.mat',
                                   {'signal_sample': sample_data})

    if data_name == 'ZXJB':

        fault_list = ['cage_crack_withoutbox',
                      'outer_crack_0.5mm_Centered_without_box',
                      'outer_crack_0.5mm_Opposite_without_box',
                      'outer_crack_0.5mm_Orthogonal_without_box',
                      'roller_crack_0.4mm_without_box',
                      'roller_crack_0.8mm_without_box',
                      'roller_crack_1.2mm_without_box',
                      'baseline_without_box']
        speed_list = ['_1000', '_1500', '_2000', '_2500', '_3000']
        load_list = ['_0_', '_20_', '_40_']

        for filepath, dirnames, filenames in os.walk(file_dirc):
            for filename in filenames:
                SourceData = pd.read_csv(os.path.join(filepath, filename))
                SourceData = SourceData.loc[:, ~SourceData.columns.str.contains('^Unnamed')]
                db = SourceData['FSy'].values
                for i, j, k in product(fault_list, speed_list, load_list):
                    if i+j+k in filename:
                        fault_name = i
                        speed_name = j
                        load_name = k
                        break
                sample_name = fault_name + speed_name + load_name
                file_num = int(filename[-5])
                if not os.path.exists(save_dirc+'\\'+speed_name+load_name):
                    os.makedirs(save_dirc+'\\'+speed_name+load_name)
                n = np.shape(db)[0]
                for i in range(n // gap - sample_num // gap - 1):
                    if i == 0:
                        sample_data = db[0:sample_num] / max(db)
                    else:
                        sample_data = db[i * gap:i * gap + sample_num] / max(db)
                    sc.savemat(os.path.join
                               (save_dirc+'\\'+speed_name+load_name,
                                sample_name + str(i + file_num * (n // gap - sample_num // gap - 1)))+ '.mat',
                               {'sample': sample_data})


    if data_name == 'ZXJG-SC':
        fault_list = ['scor', 'crac', 'lack', 'pitt', 'norm']
        speed_list = ['_1000', '_1500', '_2000']
        load_list = ['_0_', '_15_', '_30_']

        for filepath, dirnames, filenames in os.walk(file_dirc):
            for filename in filenames:
                SourceData = pd.read_csv(os.path.join(filepath, filename))
                SourceData = SourceData.loc[:, ~SourceData.columns.str.contains('^Unnamed')]
                db = SourceData['GBy'].values
                for i, j, k in fault_list, speed_list, load_list:
                    if i in filename:
                        fault_name = i
                    if j in filename:
                        speed_name = j
                    if k in filename:
                        load_name = k
                sample_name = fault_name + speed_name + load_name
                file_num = int(filename[-5])
                if not os.path.exists(save_dirc+'\\'+speed_name+load_name):
                    os.makedirs(save_dirc+'\\'+speed_name+load_name)
                n = np.shape(db)[0]
                for i in range(n // gap - sample_num // gap - 1):
                    if i == 0:
                        sample_data = db[0:sample_num] / max(db)
                    else:
                        sample_data = db[i * gap:i * gap + sample_num] / max(db)
                    sc.savemat(os.path.join
                               (save_dirc+'\\'+speed_name+load_name,
                                sample_name + str(i + file_num * (n // gap - sample_num // gap - 1)))+ '.mat',
                               {'sample': sample_data})


    elif data_name == 'WTH':
        for filepath, dirnames, filenames in os.walk(file_dirc):

            for filename in filenames:
                SourceData = sc.loadmat(os.path.join(filepath, filename))
                acc_db = SourceData['Channel_1']
                speed_db = SourceData['Channel_2']

                if 'H' in filename:
                    sample_name = 'H'
                    H_num += 1
                    file_num = H_num
                elif 'I' in filename:
                    sample_name = 'I'
                    I_num += 1
                    file_num = I_num
                elif 'O' in filename:
                    sample_name = 'O'
                    O_num += 1
                    file_num = O_num

                n = np.shape(acc_db)[0]

                for i in range(n // gap - sample_num // gap - 1):
                    if i == 0:
                        sample_acc_data = acc_db[0:sample_num] / max(acc_db)
                        sample_speed_data = speed_db[0:sample_num]
                    #     这里有一个疑问，速度数据要归一化吗？
                    else:
                        sample_acc_data = acc_db[i * gap:i * gap + sample_num] / max(acc_db)
                        sample_speed_data = speed_db[i * gap:i * gap + sample_num]

                    sc.savemat(os.path.join(save_dirc, sample_name + '_' + str(
                        i + file_num * (n // gap - sample_num // gap - 1))) + '.mat',
                               {'acc': sample_acc_data, 'speed':sample_speed_data})

    elif data_name == 'Paderborn':

        for filepath, dirnames, filenames in os.walk(file_dirc):
            for filename in filenames:
                if '.mat' in filename:
                    SourceData = sc.loadmat(os.path.join(filepath, filename))
                    acc_db = SourceData[filename[0:18]]
                    speed_db = SourceData['Channel_2']

                    if 'H' in filename:
                        sample_name = 'H'
                        H_num += 1
                        file_num = H_num
                    elif 'I' in filename:
                        sample_name = 'I'
                        I_num += 1
                        file_num = I_num
                    elif 'O' in filename:
                        sample_name = 'O'
                        O_num += 1
                        file_num = O_num

                    n = np.shape(acc_db)[0]

                    for i in range(n // gap - sample_num // gap - 1):
                        if i == 0:
                            sample_acc_data = acc_db[0:sample_num] / max(acc_db)
                            sample_speed_data = speed_db[0:sample_num]
                        #     这里有一个疑问，速度数据要归一化吗？
                        else:
                            sample_acc_data = acc_db[i * gap:i * gap + sample_num] / max(acc_db)
                            sample_speed_data = speed_db[i * gap:i * gap + sample_num]

                        sc.savemat(os.path.join(save_dirc, sample_name + '_' + str(
                            i + file_num * (n // gap - sample_num // gap - 1))) + '.mat',
                                   {'acc': sample_acc_data, 'speed':sample_speed_data})


def train_test_flit(file_dirc, rate, random=None):
    """"主要用于已有.mat数据的训练集、测试集的划分, rate是训练集比率"""
    """(误)如果样本划分是重叠的，该方法的就可能会造成数据划分不完全，训练集中包含大量测试数据！"""
    num_to_flit = 10*rate

    for filepath, dirnames, filenames in os.walk(file_dirc):
        n = 1
        for filename in filenames:
            if n > 10:
                n = 1
            elif n <= num_to_flit:
                n += 1
            elif n > num_to_flit:
                old_name = os.path.join(filepath, filename)
                new_name = os.path.join(filepath, 'test_' + filename)
                os.rename(old_name, new_name)
                n += 1
    """目前缺少一个随机划分的程序"""


def domain_sample_confusion(file_dirc):
    """主要用于s-t数据的合并"""
    for filepath, dirnames, filenames in os.walk(file_dirc):
        for condition_s in dirnames:
            for condition_t in dirnames:
                if condition_s == condition_t:
                    continue
                confuse_dir = file_dirc+'/' + condition_s + '-' + condition_t
                os.mkdir(confuse_dir)
                for _, _, filenames_list_s in os.walk(file_dirc+'/'+condition_s):
                    for filenames_s in filenames_list_s:
                        shutil.copy(file_dirc+'/'+condition_s+'/'+filenames_s, confuse_dir)
                for _, _, filenames_list_t in os.walk(file_dirc+'/'+condition_t):
                    for filenames_t in filenames_list_t:
                        shutil.copy(file_dirc+'/'+condition_t+'/'+filenames_t, confuse_dir)
                print('s:', condition_s, 't:', condition_t)


if __name__ == '__main__':

    db_dir = r'D:\DATABASE\ZXJ_test_data\fault_bearing_standard_sample\fault_sample\without_box'
    sample_dir = r'D:\DATABASE\ZXJ_test_data\fault_bearing_standard_sample\sample\without_box'
    dataslice('ZXJB', db_dir, sample_dir, 0.7)
    train_test_flit(r'D:\DATABASE\ZXJ_test_data\fault_bearing_standard_sample\sample\without_box', 0.7)
    domain_sample_confusion(sample_dir)


    # db_dir = r'D:\python_workfile\ML_Classify\samples\source_data\weak fault data'
    # sample_dir = r'D:\python_workfile\ML_Classify\samples\exp1\gap1024'
    #
    # dirname_list = ['signle_condition_1000_0', 'signle_condition_1000_20', 'signle_condition_1000_40',
    #                 'signle_condition_2000_0', 'signle_condition_2000_20', 'signle_condition_2000_40',
    #                 'signle_condition_3000_0', 'signle_condition_3000_20', 'signle_condition_3000_40',
    #                 'vary_condition_multi_load_1000', 'vary_condition_multi_load_2000', 'vary_condition_multi_load_3000',
    #                 'vary_condition_multi_speed_0', 'vary_condition_multi_speed_20', 'vary_condition_multi_speed_40']
    # for dir in dirname_list:
    #     dataslice(db_dir+"\\"+dir, sample_dir+"\\"+dir)

    # # 划分文件夹下所有子文件夹内包含的数据
    # file_fold = r'D:/DATABASE/CWRU_make_samples/samples/crack_0.021'
    # for CuDir, dirs, files in os.walk(file_fold):
    #     for i in dirs:
    #         train_test_flit('D:/DATABASE/CWRU_make_samples/samples/crack_0.007'+'/'+i, 0.7)

    # db_dir = r'D:\DATABASE\CWRU_xjs\CWRUData-original'
    # sample_dir = r'D:\DATABASE\CWRU_xjs\sample'
    # p = Pool(64)
    # for i in range(60):
    #     p.apply_async(dataslice, args=('CWRU', db_dir, sample_dir, 0.7))
    # p.close()
    # p.join()

    # domain_sample_confusion(r'D:\DATABASE\CWRU_xjs\sample\12k\Drive_End\noise\snr10\021')