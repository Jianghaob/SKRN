import time
import torch
import numpy as np
import sys

sys.path.append('../HSI_cls/')
import d2lzh_pytorch as d2l

def evaluate_accuracy(data_iter, net, loss, device):
    acc_sum, n, test_l_sum, batch_count = 0.0, 0, 0.0, 0  # 初始化为0  该行为新更改代码
    # acc_sum, n = 0.0, 0
    with torch.no_grad():  # 关闭自动求导功能，避免不必要的计算和占用内存
        for X, y in data_iter:  # X为特征  y为标签
            # test_l_sum, test_num = 0.0, 0
            X = X.to(device)
            y = y.to(device)
            net.eval()  # 评估模式, 这会关闭dropout
            y_hat = net(X)  # 将特征输入到网络中，得到预测结果
            l = loss(y_hat, y.long())  # 计算该批次样本的测试损失平均值

            l_sum = l * y.shape[0]  # 乘以批次大小  计算该批次样本的测试损失总和
            acc_sum += (y_hat.argmax(dim=1) == y.to(device)).float().sum().cpu().item()  # 旨在计算所有样本中预测正确的样本数量

            test_l_sum += l_sum  # 旨在计算所有样本的预测损失总和

            batch_count += 1  # 遍历整个数据  该batchsize迭代的次数

            net.train()  # 改回训练模式
            n += y.shape[0]  # 旨在计算样本总数量
    return [acc_sum / n, test_l_sum / n]  # 预测正确总数/所有样本数   所有样本损失函数总和/所有样本数  返回所有样本的准确率 平均损失值

def train(net, train_iter, valida_iter, loss, optimizer, device, epochs=30, early_stopping=True,
          early_num=30):
    loss_list = [100]
    early_epoch = 0

    net = net.to(device)
    print("training on ", device)
    start = time.time()
    train_loss_list = []
    valida_loss_list = []
    train_acc_list = []
    valida_acc_list = []
    for epoch in range(epochs):
        train_acc_sum, n, train_l_sum, batch_count = 0.0, 0, 0.0, 0
        time_epoch = time.time()
        lr_adjust = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 15, eta_min=0.0, last_epoch=-1)
        # lr_adjust = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=False)

        for X, y in train_iter:  # 通过batch去迭代遍历整个训练数据集
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            # print('y_hat', y_hat)
            # print('y', y)
            l = loss(y_hat, y.long())

            optimizer.zero_grad()
            l.backward()  # 损失函数反向传播
            optimizer.step()  # 梯度下降
            l_sum = l * y.shape[0]
            train_l_sum += l_sum.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]  # 记录训练样本数
            batch_count += 1  # 记录batch迭代的次数

        valida_acc, valida_loss = evaluate_accuracy(valida_iter, net, loss, device)
        lr_adjust.step(epoch)  # 更新学习率
        loss_list.append(valida_loss)

        train_loss_list.append(train_l_sum / n)
        train_acc_list.append(train_acc_sum / n)
        valida_loss_list.append(valida_loss)
        valida_acc_list.append(valida_acc)

        print('epoch %d, train loss %.6f, train acc %.3f, valida loss %.6f, valida acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, valida_loss, valida_acc,
                 time.time() - time_epoch))

        PATH = "./net_DBA.pt"
        # if loss_list[-1] <= 0.01 and valida_acc >= 0.95:
        #     torch.save(net.state_dict(), PATH)
        #     break

        # 早停设置
        if early_stopping and loss_list[-2] < loss_list[-1]:  # < 0.05) and (loss_list[-1] <= 0.05):
            if early_epoch == 0:  # and valida_acc > 0.9:
                torch.save(net.state_dict(), PATH)
            early_epoch += 1
            loss_list[-1] = loss_list[-2]
            if early_epoch == early_num:
                net.load_state_dict(torch.load(PATH))
                break
        else:
            early_epoch = 0

    # 绘图部分
    d2l.set_figsize()
    d2l.plt.figure(figsize=(8, 8.5))  # 创建大表格

    train_loss = d2l.plt.subplot(221)
    train_loss.set_title('train_loss')
    train_loss_list = torch.tensor(train_loss_list, device=device).cpu().detach().numpy()  # 新添加的一行
    d2l.plt.plot(np.linspace(1, epoch, len(train_loss_list)), train_loss_list, color='red')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train loss')

    valida_loss = d2l.plt.subplot(222)
    valida_loss.set_title('valida_loss')

    valida_loss_list = torch.tensor(valida_loss_list, device=device).cpu().detach().numpy()  # 新添加的一行
    d2l.plt.plot(np.linspace(1, epoch, len(valida_loss_list)), valida_loss_list, color='gold')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('valida loss')

    train_accuracy = d2l.plt.subplot(223)  # 2*2的表格，编号为1
    train_accuracy.set_title('train_accuracy')
    train_acc_list = torch.tensor(train_acc_list, device=device).cpu().detach().numpy()  # 新添加的一行
    d2l.plt.plot(np.linspace(1, epoch, len(train_acc_list)), train_acc_list, color='green')

    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('train_accuracy')

    valida_accuracy = d2l.plt.subplot(224)
    valida_accuracy.set_title('valida_accuracy')

    valida_acc_list = torch.tensor(valida_acc_list, device=device).cpu().detach().numpy()  # 新添加的一行
    d2l.plt.plot(np.linspace(1, epoch, len(valida_acc_list)), valida_acc_list, color='deepskyblue')
    d2l.plt.xlabel('epoch')
    d2l.plt.ylabel('valida_accuracy')

    # d2l.plt.show()
    print('epoch %d, train loss %.4f, train acc %.3f, time %.1f sec'
          % (epoch + 1, train_l_sum / n, train_acc_sum / n, time.time() - start))