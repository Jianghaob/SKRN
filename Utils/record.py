# -*- coding: utf-8 -*-
import numpy as np


def record_output(oa_ae, aa_ae, kappa_ae, element_acc_ae, training_time_ae, testing_time_ae, path):
    f = open(path, 'a')

    sentence0 = 'OAs for each iteration are:' + str(oa_ae) + '\n'
    f.write(sentence0)
    sentence1 = 'AAs for each iteration are:' + str(aa_ae) + '\n'
    f.write(sentence1)
    sentence2 = 'KAPPAs for each iteration are:' + str(kappa_ae) + '\n' + '\n'
    f.write(sentence2)

    sentence21 = 'max_OA is: ' + str(max(oa_ae)) + '  ' + 'max_AA is: ' + str(
        max(aa_ae)) + '  ' + 'max_KAPPA is: ' + str(max(kappa_ae)) + '\n'
    f.write(sentence21)


    # sentence22 = '————————————————去除OA小于0.8的值再计算平均值———————————————— ' + '\n'
    # f.write(sentence22)
    # # 去除OA小于0.8的值再计算平均值,AA KAPPA对应索引的值也被移除
    # oa_ae = np.array(oa_ae)
    # aa_ae = np.array(aa_ae)
    # kappa_ae = np.array(kappa_ae)
    # mask = oa_ae >= 0.8
    #
    # oa_ae = oa_ae[mask]
    #
    # removed_indices = np.where(~mask)[0]
    #
    # aa_ae = np.delete(aa_ae, removed_indices)
    # kappa_ae = np.delete(kappa_ae, removed_indices)





    sentence3 = 'mean_OA ± std_OA is: ' + str(np.mean(oa_ae)) + ' ± ' + str(np.std(oa_ae)) + '\n'
    f.write(sentence3)
    sentence4 = 'mean_AA ± std_AA is: ' + str(np.mean(aa_ae)) + ' ± ' + str(np.std(aa_ae)) + '\n'
    f.write(sentence4)
    sentence5 = 'mean_KAPPA ± std_KAPPA is: ' + str(np.mean(kappa_ae)) + ' ± ' + str(np.std(kappa_ae)) + '\n' + '\n'
    f.write(sentence5)
    sentence6 = 'Total average Training time is: ' + str(np.sum(training_time_ae)) + '\n'
    f.write(sentence6)
    sentence7 = 'Total average Testing time is: ' + str(np.sum(testing_time_ae)) + '\n' + '\n'
    f.write(sentence7)

    element_mean = np.mean(element_acc_ae, axis=0)
    element_std = np.std(element_acc_ae, axis=0)
    sentence8 = "Mean of all elements in confusion matrix: " + str(element_mean) + '\n'
    f.write(sentence8)
    sentence9 = "Standard deviation of all elements in confusion matrix: " + str(element_std) + '\n'
    f.write(sentence9)

    f.close()
