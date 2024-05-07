import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import scipy.io as sio
import torch
import math
from Utils import extract_samll_cubic
import torch.utils.data as Data
import matplotlib.colors as colors

def load_dataset(Dataset):
    if Dataset == 'IN':
        mat_data = sio.loadmat('datasets/Indian_pines.mat')
        mat_gt = sio.loadmat('datasets/Indian_pines_gt.mat')
        data_hsi = mat_data['indian_pines_corrected']
        gt_hsi = mat_gt['indian_pines_gt']
        TOTAL_SIZE = 10249
        VALIDATION_SPLIT = 0.97
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'UP':
        uPavia = sio.loadmat('../datasets/PaviaU.mat')
        gt_uPavia = sio.loadmat('../datasets/PaviaU_gt.mat')
        data_hsi = uPavia['paviaU']
        gt_hsi = gt_uPavia['paviaU_gt']
        TOTAL_SIZE = 42776
        VALIDATION_SPLIT = 0.997
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'PC':
        uPavia = sio.loadmat('../datasets/Pavia.mat')
        gt_uPavia = sio.loadmat('../datasets/Pavia_gt.mat')
        data_hsi = uPavia['pavia']
        gt_hsi = gt_uPavia['pavia_gt']
        TOTAL_SIZE = 148152
        VALIDATION_SPLIT = 0.999
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'SV':
        SV = sio.loadmat('../datasets/Salinas_corrected.mat')
        gt_SV = sio.loadmat('../datasets/Salinas_gt.mat')
        data_hsi = SV['salinas_corrected']
        gt_hsi = gt_SV['salinas_gt']
        TOTAL_SIZE = 54129
        VALIDATION_SPLIT = 0.997
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'KSC':
        KSC = sio.loadmat('../datasets/KSC.mat')
        gt_KSC = sio.loadmat('../datasets/KSC_gt.mat')
        data_hsi = KSC['KSC']
        gt_hsi = gt_KSC['KSC_gt']
        TOTAL_SIZE = 5211
        VALIDATION_SPLIT = 0.99
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)

    if Dataset == 'BS':
        BS = sio.loadmat('../datasets/Botswana.mat')
        gt_BS = sio.loadmat('../datasets/Botswana_gt.mat')
        data_hsi = BS['Botswana']
        gt_hsi = gt_BS['Botswana_gt']
        TOTAL_SIZE = 3248
        VALIDATION_SPLIT = 0.988
        TRAIN_SIZE = math.ceil(TOTAL_SIZE * VALIDATION_SPLIT)
    return data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT



def save_cmap(img, cmap, fname):#图像前两个是高 宽
    sizes = np.shape(img)
    height = float(sizes[0])
    width = float(sizes[1])

    fig = plt.figure()
    fig.set_size_inches(width / height, 1, forward=False) # 设置图形的高度为1英寸
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off() # 关闭坐标轴的显示
    fig.add_axes(ax)# 将创建的坐标轴添加到图形中

    ax.imshow(img, cmap=cmap)
    plt.savefig(fname, dpi=height)
    plt.close()


def sampling(proportion, ground_truth):
    train = {}
    test = {}
    labels_loc = {} #键——类，值——下标
    m = max(ground_truth)
    for i in range(m): #16
        indexes = [j for j, x in enumerate(ground_truth.ravel().tolist()) if x == i + 1] #列表，存的i类，所有的像素的下标，组成列表
        np.random.shuffle(indexes) #打乱
        labels_loc[i] = indexes
        if proportion != 1:
            nb_val = max(int((1 - proportion) * len(indexes)), 3) #至少三个
            # nb_val = max(int((1 - proportion) * len(indexes)), 1)
            # nb_val = 15
            class_num = {} #键——类，值——验证样本下标数组
            class_num[i] = indexes[:nb_val]
            print("类别", i + 1, "的训练样本数:", len(class_num[i]))

        else:
            nb_val = 0

        train[i] = indexes[:nb_val] #训练样本等于验证样本，剩下的用于测试
        test[i] = indexes[nb_val:]
    train_indexes = []
    test_indexes = []

    for i in range(m):
        train_indexes += train[i] #[000001111122222]但是是下标
        test_indexes += test[i]
    np.random.shuffle(train_indexes)
    np.random.shuffle(test_indexes)
    return train_indexes, test_indexes
'''
gt是21025的，但我只需要知道我要的样本的下标，把它返回出去就行。我先找到每类 都有哪些下标。再根据每类选训练集测试集个数
'''


def aa_and_each_accuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum)) # each_acc表示预测正确的值/预测该类的所有值
    average_acc = np.mean(each_acc)
    return each_acc, average_acc


def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0


def list_to_colormap(x_list): # 传入标签
    y = np.zeros((x_list.shape[0], 3)) # 21025,3
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([255, 0, 0]) / 255. # 有意思
        if item == 1:
            y[index] = np.array([0, 255, 0]) / 255.
        if item == 2:
            y[index] = np.array([0, 0, 255]) / 255.
        if item == 3:
            y[index] = np.array([255, 255, 0]) / 255.
        if item == 4:
            y[index] = np.array([0, 255, 255]) / 255.
        if item == 5:
            y[index] = np.array([255, 0, 255]) / 255.
        if item == 6:
            y[index] = np.array([192, 192, 192]) / 255.
        if item == 7:
            y[index] = np.array([128, 128, 128]) / 255.
        if item == 8:
            y[index] = np.array([128, 0, 0]) / 255.
        if item == 9:
            y[index] = np.array([128, 128, 0]) / 255.
        if item == 10:
            y[index] = np.array([0, 128, 0]) / 255.
        if item == 11:
            y[index] = np.array([128, 0, 128]) / 255.
        if item == 12:
            y[index] = np.array([0, 128, 128]) / 255.
        if item == 13:
            y[index] = np.array([0, 0, 128]) / 255.
        if item == 14:
            y[index] = np.array([255, 165, 0]) / 255.
        if item == 15:
            y[index] = np.array([255, 215, 0]) / 255.
        if item == 16:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 17:
            y[index] = np.array([215, 255, 0]) / 255.
        if item == 18:
            y[index] = np.array([0, 255, 215]) / 255.
        if item == -1:
            y[index] = np.array([0, 0, 0]) / 255.
    return y


def generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices, TOTAL_SIZE, total_indices, VAL_SIZE, #whole_data就是原始数据，padded_data就是只加了xy上的pad的数据
                  whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION, batch_size, gt):
    gt_all = gt[total_indices] - 1 #把所有的标签都 -1；-1的数据不要
    y_train = gt[train_indices] - 1
    y_test = gt[test_indices] - 1

    #
    all_data = extract_samll_cubic.select_small_cubic(TOTAL_SIZE, total_indices, whole_data,
                                                      PATCH_LENGTH, padded_data, INPUT_DIMENSION)

    train_data = extract_samll_cubic.select_small_cubic(TRAIN_SIZE, train_indices, whole_data,
                                                        PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    test_data = extract_samll_cubic.select_small_cubic(TEST_SIZE, test_indices, whole_data,
                                                       PATCH_LENGTH, padded_data, INPUT_DIMENSION)
    x_train = train_data.reshape(train_data.shape[0], train_data.shape[1], train_data.shape[2], INPUT_DIMENSION) # 训练集
    x_test_all = test_data.reshape(test_data.shape[0], test_data.shape[1], test_data.shape[2], INPUT_DIMENSION) # 测试集

    x_val = x_test_all[-VAL_SIZE:] #验证集
    y_val = y_test[-VAL_SIZE:]  #标签

    x_test = x_test_all[:-VAL_SIZE] #测试+标签 对测试数据划分为验证集 + 测试集
    y_test = y_test[:-VAL_SIZE]
    # print('y_train', np.unique(y_train))
    # print('y_val', np.unique(y_val))
    # print('y_test', np.unique(y_test))
    # print(y_val)
    # print(y_test)

    # K.clear_session()  # clear session before next loop

    # print(y1_train)
    # y1_train = to_categorical(y1_train)  # to one-hot labels
    # 封装成PyTorch的Dataset对象，创建一个可以被数据加载器（DataLoader）使用的轻量级的数据集
    x1_tensor_train = torch.from_numpy(x_train).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    torch_dataset_train = Data.TensorDataset(x1_tensor_train, y1_tensor_train)

    x1_tensor_valida = torch.from_numpy(x_val).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_valida = torch.from_numpy(y_val).type(torch.FloatTensor)
    torch_dataset_valida = Data.TensorDataset(x1_tensor_valida, y1_tensor_valida)

    x1_tensor_test = torch.from_numpy(x_test).type(torch.FloatTensor).unsqueeze(1)
    y1_tensor_test = torch.from_numpy(y_test).type(torch.FloatTensor)
    torch_dataset_test = Data.TensorDataset(x1_tensor_test, y1_tensor_test)

    all_data.reshape(all_data.shape[0], all_data.shape[1], all_data.shape[2], INPUT_DIMENSION)
    all_tensor_data = torch.from_numpy(all_data).type(torch.FloatTensor).unsqueeze(1)
    all_tensor_data_label = torch.from_numpy(gt_all).type(torch.FloatTensor)
    torch_dataset_all = Data.TensorDataset(all_tensor_data, all_tensor_data_label)
    # 这样我就可以使用DataLoader的功能，将样本分批Batch_size
    train_iter = Data.DataLoader(
        dataset=torch_dataset_train,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    valiada_iter = Data.DataLoader(
        dataset=torch_dataset_valida,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    test_iter = Data.DataLoader(
        dataset=torch_dataset_test,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    all_iter = Data.DataLoader(
        dataset=torch_dataset_all,  # torch TensorDataset format
        batch_size=batch_size,  # mini batch size
        shuffle=False,  # 要不要打乱数据 (打乱比较好)
        num_workers=0,  # 多线程来读数据
    )
    return train_iter, valiada_iter, test_iter, all_iter  # , y_test


def generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices): #分批数据，net网络，标签，数据集，设备，索引
    pred_test = []

    for X, y in all_iter:
        X = X.to(device)
        net.eval()  # 评估模式, 这会关闭dropout
        # print(net(X))
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))

    gt = gt_hsi.flatten() # 标签
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            # x[i] = 16
            x_label[i] = 16
        # else:
        #     x_label[i] = pred_test[label_list]
        #     label_list += 1
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label) # 预测标签

    # print('-------Save the result in mat format--------')
    # x_re = np.reshape(x, (gt_hsi.shape[0], gt_hsi.shape[1]))
    # sio.savemat('mat/' + Dataset + '_' + '.mat', {Dataset: x_re})

    y_list = list_to_colormap(x) #
    y_gt = list_to_colormap(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3)) # 转RGB图片
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    path = net.name
    classification_map(y_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_' + net.name + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_gt.png')
    print('------Get classification maps successful-------')


def generate_pseudocolor_png(all_iter, net, gt_hsi, Dataset, device, total_indices):
    pred_test = []

    for X, y in all_iter:
        X = X.to(device)
        net.eval()  # 评估模式, 这会关闭dropout
        # print(net(X))
        pred_test.extend(np.array(net(X).cpu().argmax(axis=1)))

    gt = gt_hsi.flatten()
    x_label = np.zeros(gt.shape)
    for i in range(len(gt)):
        if gt[i] == 0:
            gt[i] = 17
            # x[i] = 16
            x_label[i] = 16
        # else:
        #     x_label[i] = pred_test[label_list]
        #     label_list += 1
    gt = gt[:] - 1
    x_label[total_indices] = pred_test
    x = np.ravel(x_label)

    y_list = list_to_colormap(x)
    cmap = plt.get_cmap('jet', len(np.unique(x)))
    norm = colors.BoundaryNorm(np.arange(len(np.unique(x)) + 1) - 0.5, len(np.unique(x)))
    y_gt = cmap(norm(x))

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 4))

    path = net.name
    classification_map(y_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_' + net.name + '.png')
    classification_map(gt_re, gt_hsi, 300,
                       path + '/classification_maps/' + Dataset + '_gt_pseudocolor.png')
    print('------Get classification maps successful-------')

