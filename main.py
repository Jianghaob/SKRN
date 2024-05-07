import torch
import time
import numpy as np
import train
import os
import datetime
import collections

from torch import optim
from  sklearn import  metrics, preprocessing
from generate_pic import aa_and_each_accuracy, sampling, load_dataset, generate_png, generate_iter
from network import res2_SK_GAatt
from Utils import record, extract_samll_cubic

if not os.path.exists('./net'):
    os.makedirs('./net')
import sys


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# for Monte Carlo runs
seeds = [1331, 1332, 1333, 1334, 1335, 1336, 1337, 1338, 1339, 1340,
         1341, 1342, 1343, 1344, 1345, 1346, 1347, 1348, 1349, 1350, 1351]
ensemble = 1

day = datetime.datetime.now()
day_str = day.strftime('%m_%d_%H_%M')  # 获得当时时间

print('-----Importing Dataset-----')

global Dataset  # UP,IN,KSC...
dataset = input('Please input the name of Dataset(IN, UP, BS, SV, PC or KSC):')
Dataset = dataset.upper()  # 把输入都转换为大写，赋值为Dataset
data_hsi, gt_hsi, TOTAL_SIZE, TRAIN_SIZE, VALIDATION_SPLIT = load_dataset(Dataset)

# 加载数据集，返回数据、真实标签值、总样本数量、训练数量、训练验证集比例划分


print(data_hsi.shape)  # (145, 145, 200)
image_x, image_y, BAND = data_hsi.shape  # W,H,C (145,145,200)
data = data_hsi.reshape(np.prod(data_hsi.shape[:2]), np.prod(data_hsi.shape[2:]))  # (21025, 200)   np.prod计算乘积
gt = gt_hsi.reshape(np.prod(gt_hsi.shape[:2]), )  # (21025, )

CLASSES_NUM = max(gt)  # 1-16个数字，16最大，所以有16个类
print('The class numbers of the HSI data is:', CLASSES_NUM)

print('-----Importing Setting Parameters-----')
ITER = 10
PATCH_LENGTH = 4  # 不直接设置为9，为了方便下面添加padding操作

# number of training samples per class
num_perclass = '3%'
lr1, num_epochs1, batch_size1 = 0.00050, 10, 64
lr2, num_epochs2, batch_size2 = 0.00050, 10, 64
lr3, num_epochs3, batch_size3 = 0.00050, 10, 64
lr4, num_epochs4, batch_size4 = 0.00050, 10, 64
lr5, num_epochs5, batch_size5 = 0.00050, 10, 64
lr6, num_epochs6, batch_size6 = 0.00050, 10, 64

loss = torch.nn.CrossEntropyLoss()

img_rows = 2 * PATCH_LENGTH + 1
img_cols = 2 * PATCH_LENGTH + 1  # 输入的图像为以像素为中心，取9*9的小图
img_channels = data_hsi.shape[2]  # 9*9*200
INPUT_DIMENSION = data_hsi.shape[2]  # 输入通道数
ALL_SIZE = data_hsi.shape[0] * data_hsi.shape[1]  # 所有的像素点个数
VAL_SIZE = int(TRAIN_SIZE)  # 验证集数量与训练集数量一致
TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE  # 测试集数量
#
KAPPA1 = []
OA1 = []
AA1 = []
TRAINING_TIME1 = []
TESTING_TIME1 = []
ELEMENT_ACC1 = np.zeros((ITER, CLASSES_NUM)) # 迭代次数 * 类数：列代表16个类

KAPPA2 = []
OA2 = []
AA2 = []
TRAINING_TIME2 = []
TESTING_TIME2 = []
ELEMENT_ACC2 = np.zeros((ITER, CLASSES_NUM))

KAPPA3 = []
OA3 = []
AA3 = []
TRAINING_TIME3 = []
TESTING_TIME3 = []
ELEMENT_ACC3 = np.zeros((ITER, CLASSES_NUM))

KAPPA4 = []
OA4 = []
AA4 = []
TRAINING_TIME4 = []
TESTING_TIME4 = []
ELEMENT_ACC4 = np.zeros((ITER, CLASSES_NUM))

KAPPA5 = []
OA5 = []
AA5 = []
TRAINING_TIME5 = []
TESTING_TIME5 = []
ELEMENT_ACC5 = np.zeros((ITER, CLASSES_NUM))

KAPPA6 = []
OA6 = []
AA6 = []
TRAINING_TIME6 = []
TESTING_TIME6 = []
ELEMENT_ACC6 = np.zeros((ITER, CLASSES_NUM))

data = preprocessing.scale(data)  # 标准化处理 (21025, 200)
data_ = data.reshape(data_hsi.shape[0], data_hsi.shape[1], data_hsi.shape[2])
whole_data = data_
padded_data = np.lib.pad(whole_data, ((PATCH_LENGTH, PATCH_LENGTH), (PATCH_LENGTH, PATCH_LENGTH), (0, 0)),
                         'constant', constant_values=0)  # 添加padding操作，从145*145*200变为了153*153*200，方便在取边缘像素时，也能取9*9的大小




for index_iter in range(ITER):  #10
    net = res2_SK_GAatt(BAND, CLASSES_NUM)
    print(str(net.name) + '  iter: ' + str(index_iter + 1))
    optimizer = optim.Adam(net.parameters(), lr=lr1)  # , weight_decay=0.0001)
    time_1 = int(time.time())
    np.random.seed(seeds[index_iter])  # 这样做的目的是保证每个训练迭代中的随机数序列都是相同的，从而使得训练过程具有可重现性
    # np.random.seed(seeds[1])

    train_indices, test_indices = sampling(VALIDATION_SPLIT, gt) # 理清这步挺关键 1 - 0.97 的训练样本
    _, total_indices = sampling(1, gt)

    TRAIN_SIZE = len(train_indices) #训练大小
    print('Train size: ', TRAIN_SIZE)
    TEST_SIZE = TOTAL_SIZE - TRAIN_SIZE #测试大小=总大小-训练大小

    print('Test size: ', TEST_SIZE - TRAIN_SIZE)
    VAL_SIZE = int(TRAIN_SIZE)  #验证大小=训练大小
    print('Validation size: ', VAL_SIZE)

    print('-----Selecting Small Pieces from the Original Cube Data-----')

    # 生成分批次数据
    train_iter, valida_iter, test_iter, all_iter = generate_iter(TRAIN_SIZE, train_indices, TEST_SIZE, test_indices,
                                                                 TOTAL_SIZE, total_indices, VAL_SIZE,
                                                                 whole_data, PATCH_LENGTH, padded_data, INPUT_DIMENSION,
                                                                 batch_size1, gt)

    tic1 = time.process_time()
    # 带入训练和验证数据集
    train.train(net, train_iter, valida_iter, loss, optimizer, device, epochs=num_epochs1) #500epoch
    toc1 = time.process_time()


    def score_function(engine):
        val_loss = engine.state.metrics['nll']
        return -val_loss


    # 测试阶段



    pred_test_DBMA = []
    tic2 = time.process_time()
    with torch.no_grad():  # 临时禁用 PyTorch 自动求导功能
        for X, y in test_iter: # 遍历测试数据集
            X = X.to(device)
            net.eval()  # 评估模式, 这会关闭dropout
            y_hat = net(X)
            # print(net(X))
            pred_test_DBMA.extend(np.array(net(X).cpu().argmax(axis=1))) #返回预测最高 类别的索引
    # print('len', len(pred_test_DBMA))
    toc2 = time.process_time()
    '''
    假设1000个样本里面预测了 真实标签为1的数据预测成为了x，我统计预测后 预测类对应的次数
    '''
    collections.Counter(pred_test_DBMA)  # 输出类似Counter({'apple': 2, 'orange': 2, 'pear': 1, 'banana': 1}) 某个东西出现的次数
    gt_test = gt[test_indices] - 1

    overall_acc_DBMA = metrics.accuracy_score(pred_test_DBMA, gt_test[:-VAL_SIZE])
    confusion_matrix_DBMA = metrics.confusion_matrix(pred_test_DBMA, gt_test[:-VAL_SIZE])
    each_acc_DBMA, average_acc_DBMA = aa_and_each_accuracy(confusion_matrix_DBMA)

    kappa = metrics.cohen_kappa_score(pred_test_DBMA, gt_test[:-VAL_SIZE])

    torch.save(net.state_dict(), "./net/" + net.name + '_' + str(round(overall_acc_DBMA, 3)) + '.pt')
    KAPPA1.append(kappa)
    OA1.append(overall_acc_DBMA)
    AA1.append(average_acc_DBMA)
    TRAINING_TIME1.append(toc1 - tic1)
    TESTING_TIME1.append(toc2 - tic2)
    ELEMENT_ACC1[index_iter, :] = each_acc_DBMA #

print("--------" + net.name + " Training Finished-----------")
record.record_output(OA1, AA1, KAPPA1, ELEMENT_ACC1, TRAINING_TIME1, TESTING_TIME1,

                     'records/' + day_str + '_1' + Dataset + 'split：' + str(num_perclass) + 'perclass' + net.name + '_' + 'lr：' + str(
                         lr1) + 'batchsize：' + str(batch_size1) + '.txt')

generate_png(all_iter, net, gt_hsi, Dataset, device, total_indices)
