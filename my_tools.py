import os, pickle, csv
import numpy as np
from timeit import default_timer as timer
from sklearn import metrics as metrics
import shutil


def now():
    from datetime import datetime
    return datetime.now()


def del_dir(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def format_path(data_path, Itype='.p'):
    '''补充后缀如.p，生成对应目录避免错误'''
    if '.' not in data_path or data_path.split('.')[-1] != Itype[1:]:
        print('后缀有误，已自动添加{}'.format(Itype))
        data_path += Itype
    folder_path = os.path.dirname(data_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return data_path


def cache_data(data, data_path):
    '''.p存'''
    data_path = format_path(data_path, Itype='.p')

    with open(data_path, 'wb') as f:
        pickle.dump(data, f)
    print('Done cache_data for {}.'.format(data_path))


def load_cached_data(data_path):
    '''.p读'''
    data_path = format_path(data_path, Itype='.p')

    with open(data_path, 'rb') as f:
        model = pickle.load(f)
    print('Done load_cached_data for {}.'.format(data_path))
    return model


def to_csv(pred, csv_path):
    '''to_csv与read_csv配套，用于list或者是nparray'''
    csv_path = format_path(csv_path, Itype='.csv')
    if (type(pred[0]) != type([])) and (type(pred[0]) != type(np.array([]))):
        pred = np.array(pred).reshape(-1, 1)
    # pred = np.array(pred).reshape(-1, col_num)

    with open(csv_path, 'w', newline='') as f:
        f_csv_writer = csv.writer(f)

        f_csv_writer.writerows(pred)
        f.close()


def read_csv(csv_path):
    '''to_csv与read_csv配套，用于list或者是nparray'''
    csv_path = format_path(csv_path, Itype='.csv')
    RMSEs = []
    one_array_flag = True
    with open(csv_path, 'r') as f:
        for csv_row in f:
            i_RMSEs = []
            for i_data in csv_row.split(','):
                i_RMSEs.append(float(i_data))
            if one_array_flag:
                if len(i_RMSEs) == 1:
                    RMSEs.extend(i_RMSEs)
                else:
                    one_array_flag = False
                    new_RMSEs = []
                    for q in RMSEs:
                        new_RMSEs.append(np.array([q]))
                    RMSEs = new_RMSEs
            if not one_array_flag:
                RMSEs.append(np.array(i_RMSEs))
        f.close()
    return np.array(RMSEs)


def set_random_seed(seed=42, deterministic=True):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except:
        pass


def evaluate_true_pred_label(y_true, y_pred, desc='', para='strong'):
    '''根据true和pred评估结果，true在前'''
    try:
        num = y_true.shape[0]
    except:
        num = len(y_true)
    if num == 0:
        return

    print('-' * 10 + desc + '-' * 10)
    cf_flow = metrics.confusion_matrix(y_true, y_pred)
    if len(cf_flow.ravel()) == 1:
        if y_true[0] == 0:
            TN, FP, FN, TP = cf_flow[0][0], 0, 0, 0
        elif y_true[0] == 1:
            TN, FP, FN, TP = 0, 0, 0, cf_flow[0][0]
        else:
            raise Exception("label error")
    else:
        TN, FP, FN, TP = cf_flow.ravel()

    rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
    Accu = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0

    F1 = 2 * rec * prec / (rec + prec) if (rec + prec) != 0 else 0
    if para.lower() == 'strong'.lower():
        print("TP:\t" + str(TP), end='\t|| ')
        print("FP:\t" + str(FP), end='\t|| ')
        print("TN:\t" + str(TN), end='\t|| ')
        print("FN:\t" + str(FN))
        print("Recall:\t{:6.4f}".format(rec), end='\t|| ')
        print("Precision:\t{:6.4f}".format(prec))
        print("Accuracy:\t{:6.4f}".format(Accu), end='\t|| ')
        print("F1:\t{:6.4f}".format(F1))
    else:
        print("\tTP \t" + str(TP), end='\t - ')
        print("\tFP \t" + str(FP), end='\t - ')
        print("\tTN \t" + str(TN), end='\t - ')
        print("\tFN \t" + str(FN))
        print("\tRecall \t{:6.4f}".format(rec), end='\t - ')
        print("\tPrecision \t{:6.4f}".format(prec))
        print("\tAccuracy \t{:6.4f}".format(Accu), end='\t - ')
        print("\tF1 \t{:6.4f}".format(F1))


def search_for_OWAD_thres(rmse, y_true, thres, epoch=0):
    y_pred = []
    for i_r in rmse:
        if i_r < thres:
            y_pred.append(0)
        else:
            y_pred.append(1)

    cf_flow = metrics.confusion_matrix(y_true, y_pred)
    if len(cf_flow.ravel()) == 1:
        if y_true[0] == 0:
            TN, FP, FN, TP = cf_flow[0][0], 0, 0, 0
        elif y_true[0] == 1:
            TN, FP, FN, TP = 0, 0, 0, cf_flow[0][0]
        else:
            raise Exception("label error")
    else:
        TN, FP, FN, TP = cf_flow.ravel()
    rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
    score2 = (TN / (TN + FP)) if (TN + FP) != 0 else 0

    if np.abs(rec - score2) < 0.1 + epoch * 0.01 and np.abs(rec - prec) < 0.1 + epoch * 0.01:
        print('')
        return thres
    elif rec > score2:
        print(end='↑')
        return search_for_OWAD_thres(rmse, y_true, thres * 11 / 9, epoch + 1)
    else:
        print(end='↓')
        return search_for_OWAD_thres(rmse, y_true, thres * 8 / 9, epoch + 1)


# def cluster(test_x, test_y):
#     '''伪标签算法，输入ndarray，输出ndarray'''
#     set_random_seed()
#     import pandas as pd
#     from sklearn.cluster import KMeans
#     from sklearn.decomposition import PCA
#
#     test_x = pd.DataFrame(test_x)
#     test_y = pd.DataFrame(test_y, columns=[' Label'])
#     data = pd.concat([test_x, test_y], axis=1)
#     data = data.reset_index(drop=True)
#     X = data.iloc[:, :data.shape[1] - 1].values
#     # y = data.iloc[:, data.shape[1] - 1].values
#
#     print('-' * 10 + 'Try fake labeling...' + '-' * 10)
#
#     # clusters = np.arange(2, 8)
#     clusters = np.arange(2, 10)
#
#     parameter = np.arange(2, 10)
#     bestK = 0
#     best_sih = 0
#     best_pca = 0
#     tempX = X
#     for j in parameter:
#         # print(tempX.shape)
#         pca = PCA(n_components=j)
#         tempX = pca.fit_transform(tempX)
#         for i in clusters:
#             model = KMeans(n_clusters=i, n_init=10)
#             # 模型拟合
#             model.fit(tempX)
#             # 为每个示例分配一个集群
#             yhat = model.predict(tempX)
#             sih = metrics.silhouette_score(tempX, yhat)
#             if sih >= best_sih:
#                 print('[-] new best at: parameter:{:4d}, clusters:{:4d}, sih:{}'.format(j, i, sih))
#                 best_sih = sih
#                 bestK = i
#                 best_pca = j
#         tempX = X
#
#     print('Applying labeling...')
#     pca = PCA(n_components=best_pca)
#     tempX = pca.fit_transform(X)
#     model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
#     model.fit(tempX)
#     # 为每个示例分配一个集群
#     yhat = model.predict(tempX)
#     # 检索唯一群集
#     clusters = np.unique(yhat)
#     # 为每个聚类赋标签
#     for cluster in clusters:
#         # 获取此群集的示例的行索引
#         row_ix = np.where(yhat == cluster)
#         row = np.hstack(np.ravel(row_ix))
#         n = np.random.choice(row, size=20, replace=True)
#
#         if sum(data.loc[n, ' Label']) >= 10:
#             data.loc[row, ' Label'] = 1
#         else:
#             data.loc[row, ' Label'] = 0
#
#     xx, yy = data.iloc[:, :data.shape[1] - 1].values, data.iloc[:, data.shape[1] - 1].values
#     print('-' * 10 + 'Done labeling.' + '-' * 10)
#     return xx, yy
#
#
# def cluster3(test_x, test_y, label_num=200, clf='KMeans', PCA_parameter=0.99, max_cluster_num=8):
#     '''伪标签算法，输入ndarray，输出ndarray'''
#     set_random_seed()
#     from sklearn.decomposition import PCA
#     import copy
#
#     # PCA_parameter = 'mle'
#     PCA_parameter = 0.99
#     # parameter = 2
#     max_cluster_num = 8
#
#     Active_size = int(label_num * 0.2)
#     clusters = np.arange(2, max_cluster_num)
#     # # DEBUG
#     # Active_size = 200
#     # max_cluster_num = 10
#     # single_cluster_num = 20
#
#     print('-' * 10 + 'Try fake labeling...' + '-' * 10)
#     split_test_x = copy.copy(test_x[Active_size:])
#     split_test_y1 = np.array(test_y[:Active_size])
#     split_test_y2 = np.array(test_y[Active_size:])
#     if len(split_test_x) <= 0:
#         return test_x, split_test_y1
#
#     bestK, best_sih = 0, 0
#     tempX = split_test_x
#     # for j in parameter:
#     if True:
#         # print(tempX.shape)
#         pca = PCA(n_components=PCA_parameter)
#
#         try:  # 此处可能出现数学域错误
#             tempX = pca.fit_transform(tempX)
#         except:
#             split_test_x = np.array(split_test_x) + 1e-99
#             tempX = split_test_x
#             tempX = pca.fit_transform(tempX)
#
#         for i_cluster in clusters:
#             if clf.lower() == 'KMeans'.lower():
#                 from sklearn.cluster import KMeans
#                 model = KMeans(n_clusters=i_cluster, n_init=10)
#             elif clf.lower() == 'GaussianMixture'.lower():
#                 from sklearn.mixture import GaussianMixture
#                 model = GaussianMixture(n_components=i_cluster, covariance_type='full', random_state=0)
#             else:
#                 model = None
#             # 模型拟合
#             model.fit(tempX)
#             # 为每个示例分配一个集群
#             yhat = model.predict(tempX)
#             sih = metrics.silhouette_score(tempX, yhat)
#             if sih >= best_sih:
#                 print('[-] new best at: {} :{:4d}, clusters :{:4d}, sih:{}'.format(
#                     PCA_parameter, len(tempX[0]), i_cluster, sih))
#                 best_sih = sih
#                 bestK = i_cluster
#
#     print('Applying labeling...')
#     pca = PCA(n_components=PCA_parameter)
#     tempX = pca.fit_transform(split_test_x)
#
#     if clf.lower() == 'KMeans'.lower():
#         from sklearn.cluster import KMeans
#         model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
#         # model = KMeans(n_clusters=bestK, n_init=10)
#     elif clf.lower() == 'GaussianMixture'.lower():
#         from sklearn.mixture import GaussianMixture
#         model = GaussianMixture(n_components=bestK, covariance_type='full', random_state=0)
#     else:
#         model = None
#     model.fit(tempX)
#     # 为每个示例分配一个集群
#     yhat = model.predict(tempX)
#     # 检索唯一群集
#     clusters = np.unique(yhat)
#
#     single_cluster_num = int(label_num * 0.8) // len(clusters)
#     # 为每个聚类赋标签
#     yy = np.array([0] * len(split_test_y2))
#     for cluster in clusters:
#         # 获取此群集的示例的行索引
#         row_ix = np.where(yhat == cluster)
#         row = np.hstack(np.ravel(row_ix))
#         if len(row) <= single_cluster_num:
#             yy[row] = split_test_y2[row]
#         else:
#             n = np.random.choice(row, size=single_cluster_num, replace=False)
#
#             yy[row] = 1 if sum(split_test_y2[n]) >= single_cluster_num // 2 else 0
#     print('-' * 10 + 'Done labeling.' + '-' * 10)
#     return test_x, np.append(split_test_y1, yy)
#
#
# def Generate_cluster_strutrue(test_x, anom_score, label_num=200, ACTIVE_WEIGHT=0.2,
#                               unique_label=2, window_size=10000,
#                               PCA_parameter=0.99, clf='KMeans', max_cluster_num=8):
#     #######################################################
#     # 1. Start Acive_learning turn
#     #######################################################
#     # 1 主动学习参数
#     ACTIVE_WEIGHT = max(min(ACTIVE_WEIGHT, 1), 0)
#     Active_size = int(label_num * ACTIVE_WEIGHT)
#     print('[*] Step 1.1: Active_size: {} / {}'.format(Active_size, label_num))
#     # if len(test_x) <= Active_size or label_num - Active_size <= 0:
#     #     print('[*] Lack of samples, no need to cluster samples.')
#     #     print('\t' + '-' * 10 + 'Done Active Turn.' + '-' * 10)
#     #     return Active_idxs, [[Active_size, Active_idxs]]
#     #     # return test_x, Active_idxs
#
#     if len(test_x) <= label_num:
#         total_idxs = np.arange(len(test_x))
#         print('[*] Drift samples less than label_num, no need to cluster samples.')
#         print('\t' + '-' * 10 + 'Done Mixed Turn.' + '-' * 10)
#         return total_idxs, [[len(test_x), total_idxs]]
#
#     Active_idxs = np.arange(min(len(test_x), Active_size))
#     #######################################################
#     # 2.1 Start try Clustering turn
#     #######################################################
#     print('[*] Step 1.2: Cluster_size: {} / {}'.format(label_num - Active_size, label_num))
#     print('-' * 10 + 'Try fake labeling...' + '-' * 10)
#     import copy
#
#     split_test_x2 = copy.copy(test_x[Active_size:])
#
#     set_random_seed(1)
#     bestK, best_sih = 0, 0
#     Trying_Clusters_List = np.arange(2, max_cluster_num)
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=PCA_parameter)
#     try:  # 此处可能出现数学域错误，需要修改pca.py文件
#         PCA_formed_X = pca.fit_transform(split_test_x2)
#     except:
#         split_test_x2 = np.array(split_test_x2) + 1e-99
#         PCA_formed_X = pca.fit_transform(split_test_x2)
#     print('[-] PCA_parameter: {} , need_feature_len: {:4d}'.format(PCA_parameter, len(PCA_formed_X[0])))
#
#     for i_cluster in Trying_Clusters_List:
#         if clf.lower() == 'KMeans'.lower():
#             from sklearn.cluster import KMeans
#
#             model = KMeans(n_clusters=i_cluster, n_init=10)
#         elif clf.lower() == 'GaussianMixture'.lower() or clf.lower() == 'GMM'.lower():
#             from sklearn.mixture import GaussianMixture
#
#             model = GaussianMixture(n_components=i_cluster, covariance_type='full', random_state=0)
#         else:
#             model = None
#
#         # 模型拟合
#         model.fit(PCA_formed_X)
#
#         # 为每个示例分配一个集群
#         PCA_formed_yhat = model.predict(PCA_formed_X)
#         sih = metrics.silhouette_score(PCA_formed_X, PCA_formed_yhat)
#
#         if sih >= best_sih:
#             print('[-] new best at: clusters_num :{:4d}, sih:{}'.format(i_cluster, sih))
#             best_sih = sih
#             bestK = i_cluster
#
#     #######################################################
#     # 2.2 Decide Clustering_model, Fit and Cluster
#     #######################################################
#     set_random_seed(2)
#     print('Applying cluster_num...')
#
#     if clf.lower() == 'KMeans'.lower():
#         from sklearn.cluster import KMeans
#
#         model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
#         # or by the following
#         # model = KMeans(n_clusters=bestK, n_init=10)
#     elif clf.lower() == 'GaussianMixture'.lower() or clf.lower() == 'GMM'.lower():
#         from sklearn.mixture import GaussianMixture
#
#         model = GaussianMixture(n_components=bestK, covariance_type='full', random_state=0)
#     else:
#         model = None
#     model.fit(PCA_formed_X)
#     # 为每个示例分配一个集群
#     PCA_formed_yhat = model.predict(PCA_formed_X)
#     # 检索唯一群集
#     clusters = np.unique(PCA_formed_yhat)
#
#     all_cluster_label_num = int(label_num - Active_size)
#     needed_label_idx = [Active_idxs]
#     sample_idxs_list = [[Active_size, Active_idxs]]
#     #######################################################
#     # 2.3 Start Sampling and Pred
#     #######################################################
#     set_random_seed(3)
#     way = 'MODE_2'
#
#     if way == 'MODE_1':
#         # __MODE 1__: balanced cluster_label
#         single_cluster_label_num = all_cluster_label_num // len(clusters)
#         for i_cluster in clusters:
#             # 获取此群集的示例的行索引
#             clustered_sample_idxs = np.where(PCA_formed_yhat == i_cluster)[0] + Active_size
#             if len(clustered_sample_idxs) <= single_cluster_label_num:
#                 needed_label_idx.append(clustered_sample_idxs)
#
#                 sample_idxs_list.append([1, clustered_sample_idxs])
#             else:
#                 random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)),
#                                               size=single_cluster_label_num, replace=False)
#                 needed_label_idx.append(clustered_sample_idxs[random_idx])
#
#                 sample_idxs_list.append([single_cluster_label_num, clustered_sample_idxs])
#
#     elif way == 'MODE_2':
#         # __MODE 2__: im-balanced cluster_label
#         total_len = len(split_test_x2)
#         for i_cluster in clusters:
#             # 获取此群集的示例的行索引
#             clustered_sample_idxs = np.where(PCA_formed_yhat == i_cluster)[0] + Active_size
#             if len(clustered_sample_idxs) * all_cluster_label_num <= total_len:
#                 # 当前组过小，但至少需分配一个标签
#                 total_len -= len(clustered_sample_idxs)
#                 all_cluster_label_num -= 1
#
#                 random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)), size=1, replace=False)
#                 needed_label_idx.append(clustered_sample_idxs[random_idx])
#
#                 sample_idxs_list.append([1, clustered_sample_idxs])
#             else:
#                 needed_label_idx.append(None)
#                 sample_idxs_list.append([None, clustered_sample_idxs])
#
#         for i_idx, [Num, clustered_sample_idxs] in enumerate(sample_idxs_list):
#             if Num is not None: continue
#             # Num未设定，则需要确定需要的标签数，以及需要的样本下标及对应聚类
#             # 只为每个个数足够多的类采样并分配标签
#             get_label_num = int(len(clustered_sample_idxs) * all_cluster_label_num / total_len)
#             random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)),
#                                           size=min(get_label_num, len(clustered_sample_idxs)), replace=False)
#             needed_label_idx[i_idx] = clustered_sample_idxs[random_idx]
#             sample_idxs_list[i_idx][0] = get_label_num
#     else:
#         print('[***] Not know how to decide sampling.')
#         return -1
#
#     print('\t' + '-' * 10 + 'Done Applying cluster_num.' + '-' * 10)
#     needed_label_idx_ravel = []
#     for t in needed_label_idx:
#         needed_label_idx_ravel.extend(t)
#     return np.array(needed_label_idx_ravel), sample_idxs_list
#
#
# def Generate_cluster_strutrue2(test_x, anom_score, Active_size, MIN_CLUSTER_SIZE, MIN_SIH=0.5,
#                                unique_label=2, window_size=10000,
#                                PCA_parameter=0.99, clf='KMeans', max_cluster_num=8):
#     #######################################################
#     # 1. Start Acive_learning turn
#     #######################################################
#     # 1 主动学习参数
#     if len(test_x) <= Active_size + MIN_CLUSTER_SIZE:
#         print('[*] Drift samples less than label_num, no need to cluster samples.')
#         total_idxs = np.arange(len(test_x))
#         print('\t' + '-' * 10 + 'Done Mixed Turn.' + '-' * 10)
#         return total_idxs, [[len(test_x), total_idxs]]
#
#     Active_idxs = np.arange(Active_size)
#     #######################################################
#     # 2.1 Start try Clustering turn
#     #######################################################
#     print('-' * 10 + 'Try fake labeling...' + '-' * 10)
#     import copy
#
#     split_test_x2 = copy.copy(test_x[Active_size:])
#
#     set_random_seed(1)
#     bestK, best_sih = 0, 0
#     Trying_Clusters_List = np.arange(2, max_cluster_num)
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=PCA_parameter)
#     try:  # 此处可能出现数学域错误，需要修改pca.py文件
#         PCA_formed_X = pca.fit_transform(split_test_x2)
#     except:
#         split_test_x2 = np.array(split_test_x2) + 1e-99
#         PCA_formed_X = pca.fit_transform(split_test_x2)
#     print('[-] PCA_parameter: {} , need_feature_len: {:4d}'.format(PCA_parameter, len(PCA_formed_X[0])))
#
#     for i_cluster in Trying_Clusters_List:
#         if clf.lower() == 'KMeans'.lower():
#             from sklearn.cluster import KMeans
#
#             model = KMeans(n_clusters=i_cluster, n_init=10)
#         elif clf.lower() == 'GaussianMixture'.lower() or clf.lower() == 'GMM'.lower():
#             from sklearn.mixture import GaussianMixture
#
#             model = GaussianMixture(n_components=i_cluster, covariance_type='full', random_state=0)
#         else:
#             model = None
#
#         # 模型拟合
#         model.fit(PCA_formed_X)
#
#         # 为每个示例分配一个集群
#         PCA_formed_yhat = model.predict(PCA_formed_X)
#         sih = metrics.silhouette_score(PCA_formed_X, PCA_formed_yhat)
#
#         if sih >= best_sih:
#             print('[-] new best at: clusters_num :{:4d}, sih:{}'.format(i_cluster, sih))
#             best_sih = sih
#             bestK = i_cluster
#
#     if best_sih < MIN_SIH:
#         print('cluster sih_result: {} <= {}, abandon clustering'.format(best_sih, MIN_SIH))
#         return Active_idxs, [[Active_size, Active_idxs]]
#
#     #######################################################
#     # 2.2 Decide Clustering_model, Fit and Cluster
#     #######################################################
#     set_random_seed(2)
#     print('Applying cluster_num...')
#
#     if clf.lower() == 'KMeans'.lower():
#         from sklearn.cluster import KMeans
#
#         model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
#         # or by the following
#         # model = KMeans(n_clusters=bestK, n_init=10)
#     elif clf.lower() == 'GaussianMixture'.lower() or clf.lower() == 'GMM'.lower():
#         from sklearn.mixture import GaussianMixture
#
#         model = GaussianMixture(n_components=bestK, covariance_type='full', random_state=0)
#     else:
#         model = None
#     model.fit(PCA_formed_X)
#     # 为每个示例分配一个集群
#     PCA_formed_yhat = model.predict(PCA_formed_X)
#     # 检索唯一群集
#     clusters = np.unique(PCA_formed_yhat)
#
#     all_cluster_label_num = int(MIN_CLUSTER_SIZE / best_sih)
#     needed_label_idx = [Active_idxs]
#     sample_idxs_list = [[Active_size, Active_idxs]]
#     #######################################################
#     # 2.3 Start Sampling and Pred
#     #######################################################
#     set_random_seed(3)
#     way = 'MODE_2'
#
#     if way == 'MODE_1':
#         # __MODE 1__: balanced cluster_label
#         single_cluster_label_num = all_cluster_label_num // len(clusters)
#         for i_cluster in clusters:
#             # 获取此群集的示例的行索引
#             clustered_sample_idxs = np.where(PCA_formed_yhat == i_cluster)[0] + Active_size
#             if len(clustered_sample_idxs) <= single_cluster_label_num:
#                 needed_label_idx.append(clustered_sample_idxs)
#
#                 sample_idxs_list.append([1, clustered_sample_idxs])
#             else:
#                 random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)),
#                                               size=single_cluster_label_num, replace=False)
#                 needed_label_idx.append(clustered_sample_idxs[random_idx])
#
#                 sample_idxs_list.append([single_cluster_label_num, clustered_sample_idxs])
#
#     elif way == 'MODE_2':
#         # __MODE 2__: im-balanced cluster_label
#         total_len = len(split_test_x2)
#         for i_cluster in clusters:
#             # 获取此群集的示例的行索引
#             clustered_sample_idxs = np.where(PCA_formed_yhat == i_cluster)[0] + Active_size
#             if len(clustered_sample_idxs) * all_cluster_label_num <= total_len:
#                 # 当前组过小，但至少需分配一个标签
#                 total_len -= len(clustered_sample_idxs)
#                 all_cluster_label_num -= 1
#
#                 random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)), size=1, replace=False)
#                 needed_label_idx.append(clustered_sample_idxs[random_idx])
#
#                 sample_idxs_list.append([1, clustered_sample_idxs])
#             else:
#                 needed_label_idx.append(None)
#                 sample_idxs_list.append([None, clustered_sample_idxs])
#
#         for i_idx, [Num, clustered_sample_idxs] in enumerate(sample_idxs_list):
#             if Num is not None: continue
#             # Num未设定，则需要确定需要的标签数，以及需要的样本下标及对应聚类
#             # 只为每个个数足够多的类采样并分配标签
#             get_label_num = int(len(clustered_sample_idxs) * all_cluster_label_num / total_len)
#             random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)),
#                                           size=min(get_label_num, len(clustered_sample_idxs)), replace=False)
#             needed_label_idx[i_idx] = clustered_sample_idxs[random_idx]
#             sample_idxs_list[i_idx][0] = get_label_num
#     else:
#         print('[***] Not know how to decide sampling.')
#         return -1
#
#     print('\t' + '-' * 10 + 'Done Applying cluster_num.' + '-' * 10)
#     needed_label_idx_ravel = []
#     for t in needed_label_idx:
#         needed_label_idx_ravel.extend(t)
#     return np.array(needed_label_idx_ravel), sample_idxs_list
#
#
# def Explain_and_Label_cluster_strutrue(test_X, yy_list, needed_label_idx, sample_idxs_list,
#                                        Pred_bias=0.5, CLUSTER_improve_FLAG=True,
#                                        _true_y_test=None, reli_thres=0.6):
#     print('\t' + '-' * 10 + 'Start Try_Labeling...' + '-' * 10)
#     Processed_Label_idx = 0
#     pred_y = np.array([-1] * len(test_X))
#     keep_mask = np.array([False] * len(test_X))
#     for i_idx, [Num, clustered_sample_idxs] in enumerate(sample_idxs_list):
#         # 主动学习
#         if i_idx == 0:
#             temp_yy = yy_list[Processed_Label_idx:Processed_Label_idx + Num]
#             for temp_y in temp_yy:
#                 pred_y[needed_label_idx[Processed_Label_idx]] = temp_y
#                 keep_mask[needed_label_idx[Processed_Label_idx]] = True
#                 Processed_Label_idx += 1
#         # 聚类猜测
#         else:
#             temp_yy = yy_list[Processed_Label_idx:Processed_Label_idx + Num]
#             Anom_Score = np.sum(temp_yy) / len(temp_yy)
#             if Anom_Score < Pred_bias:
#                 reliablity = (Pred_bias - Anom_Score) / (Pred_bias + 1e-99)
#             else:
#                 reliablity = (Anom_Score - Pred_bias) / (1 - Pred_bias + 1e-99)
#
#             if reliablity <= reli_thres:
#                 print('[*] Found cluster_{} NOT reliable, give-up clustering...'.format(i_idx))
#
#                 # 真标签复用
#                 if CLUSTER_improve_FLAG:
#                     for temp_y in temp_yy:
#                         pred_y[needed_label_idx[Processed_Label_idx]] = temp_y
#                         keep_mask[needed_label_idx[Processed_Label_idx]] = True
#                         Processed_Label_idx += 1
#
#             else:
#                 my_label = 1 if Anom_Score >= Pred_bias else 0
#                 # 聚类伪标签
#                 pred_y[clustered_sample_idxs] = my_label
#                 keep_mask[clustered_sample_idxs] = True
#
#                 # 真标签复用
#                 if CLUSTER_improve_FLAG:
#                     revise_true_NUM = 0
#                     for temp_y in temp_yy:
#                         if pred_y[needed_label_idx[Processed_Label_idx]] != temp_y:
#                             revise_true_NUM += 1
#                             pred_y[needed_label_idx[Processed_Label_idx]] = temp_y
#                             keep_mask[needed_label_idx[Processed_Label_idx]] = True
#                         Processed_Label_idx += 1
#                     # if revise_true_NUM>0:
#                     print('cluster {}: 真标签复位数为{}/ {}.'.format(i_idx, revise_true_NUM, Num))
#
#     if _true_y_test is not None:
#         for i_idx, [Num, clustered_sample_idxs] in enumerate(sample_idxs_list):
#             temp_keep = keep_mask[clustered_sample_idxs]
#             temp_true_y_test = np.array(_true_y_test)[clustered_sample_idxs]
#             temp_pred_y = np.array(pred_y)[clustered_sample_idxs]
#             evaluate_true_pred_label(temp_true_y_test[temp_keep],
#                                      temp_pred_y[temp_keep],
#                                      'cluster {}:'.format(i_idx), 'weak')
#
#     print('-' * 10 + 'Done labeling.' + '-' * 10)
#     return np.array(test_X)[keep_mask], np.array(pred_y)[keep_mask], keep_mask
#
#
# def Mixed_Acitive_Cluster(X_update, y_update, anom_score, pred_update,
#                           update_flag, total_label_num=200, slight_label_num=5, Pred_bias=0.5,
#                           ACTIVE_WEIGHT=0.2, CLUSTER_improve_FLAG=True):
#     my_X_update, my_y_update = [], []
#     keep_mask = np.array([])
#
#     if update_flag:
#
#         # 0. Prepare Turn
#         print('[*] Step 1: Mixed_Acitive_Cluster NUM: {}'.format(total_label_num))
#         if type(ACTIVE_WEIGHT) == type('') and ACTIVE_WEIGHT.lower() == 'AUTO'.lower():
#             # 静态设置
#             # ACTIVE_WEIGHT = 0.2
#
#             # 动态决定
#             ACTIVE_WEIGHT_0 = 0.4
#             ACTIVE_SIZE_0 = int(ACTIVE_WEIGHT_0 * total_label_num)
#             fig_X, fig_y = np.arange(ACTIVE_SIZE_0), anom_score[:ACTIVE_SIZE_0]
#             fun_y = np.poly1d(np.polyfit(fig_X, fig_y, 1))  # 组合方程
#             K_40 = np.abs(fun_y(1) - fun_y(0)) + 1e-99  # 拟合后y值
#
#             fig_X, fig_y = np.arange(total_label_num), anom_score[:total_label_num]
#             fun_y = np.poly1d(np.polyfit(fig_X, fig_y, 1))  # 组合方程
#             K_200 = np.abs(fun_y(1) - fun_y(0)) + 1e-99  # 拟合后y值
#
#             # ACTIVE_WEIGHT = 0.2 * (1 - np.exp(-K_0 / K_40)) / (1 - np.exp(-1)) * np.exp(-K_40)
#             ACTIVE_WEIGHT = ACTIVE_WEIGHT_0 * (1 - np.exp(-K_200 / K_40)) / (1 - np.exp(-1)) * np.exp(-K_200)
#
#         ACTIVE_WEIGHT = max(min(ACTIVE_WEIGHT, 1), 0)
#
#         # 1.Start Generate Cluster_Structure
#         needed_label_idx, sample_idxs_list = \
#             Generate_cluster_strutrue(X_update, anom_score, total_label_num, ACTIVE_WEIGHT)
#
#         # use needed true_label for labeling
#         y_update_needed = y_update[needed_label_idx]
#
#         my_X_update, my_y_update, keep_mask = \
#             Explain_and_Label_cluster_strutrue(X_update, y_update_needed,
#                                                needed_label_idx, sample_idxs_list,
#                                                Pred_bias, CLUSTER_improve_FLAG, y_update)
#
#     return my_X_update, my_y_update, update_flag, keep_mask
#
#
# def Mixed_Acitive_Cluster2(X_update, y_update, anom_score,
#                            update_flag, total_label_num, Pred_bias=0.5,
#                            ACTIVE_WEIGHT='auto', MIN_CLUSTER_WEIGHT=0.2, MIN_SIH=0.5, CLUSTER_improve_FLAG=True):
#     my_X_update, my_y_update = [], []
#     keep_mask = np.array([])
#
#     if update_flag:
#
#         # 0. Prepare Turn
#         print('[*] Step 1: Mixed_Acitive_Cluster MAX_NUM: {}'.format(total_label_num))
#         if type(ACTIVE_WEIGHT) == type('') and ACTIVE_WEIGHT.lower() == 'AUTO'.lower():
#             # 静态设置
#             # ACTIVE_WEIGHT = 0.2
#
#             # 动态决定
#             ACTIVE_WEIGHT_0 = 0.4
#             ACTIVE_SIZE_0 = int(ACTIVE_WEIGHT_0 * total_label_num)
#             fig_X, fig_y = np.arange(ACTIVE_SIZE_0), anom_score[:ACTIVE_SIZE_0]
#             fun_y = np.poly1d(np.polyfit(fig_X, fig_y, 1))  # 组合方程
#             K_40 = np.abs(fun_y(1) - fun_y(0)) + 1e-99  # 拟合后y值
#
#             fig_X, fig_y = np.arange(total_label_num), anom_score[:total_label_num]
#             fun_y = np.poly1d(np.polyfit(fig_X, fig_y, 1))  # 组合方程
#             K_200 = np.abs(fun_y(1) - fun_y(0)) + 1e-99  # 拟合后y值
#
#             # ACTIVE_WEIGHT = 0.2 * (1 - np.exp(-K_0 / K_40)) / (1 - np.exp(-1)) * np.exp(-K_40)
#             ACTIVE_WEIGHT = ACTIVE_WEIGHT_0 * (1 - np.exp(-K_200 / K_40)) / (1 - np.exp(-1)) * np.exp(-K_200)
#
#         ACTIVE_SIZE = int(max(min(ACTIVE_WEIGHT, 1), 0.05) * total_label_num)
#         MIN_CLUSTER_SIZE = int(max(min(MIN_CLUSTER_WEIGHT, 1), 0.05) * total_label_num)
#         MIN_SIH = max(min(MIN_SIH, 1), 0.4)
#         if ACTIVE_SIZE + MIN_CLUSTER_SIZE / MIN_SIH > total_label_num:
#             print('Pre_Set MAX_Label to big, reset MIN_CLUSTER_SIZE...')
#             MIN_CLUSTER_SIZE = (total_label_num - ACTIVE_SIZE) * MIN_SIH
#
#         # 1.Start Generate Cluster_Structure
#         needed_label_idx, sample_idxs_list = \
#             Generate_cluster_strutrue2(X_update, anom_score, ACTIVE_SIZE, MIN_CLUSTER_SIZE, MIN_SIH)
#
#         print('[*****] Confirm Mixed_Act_Clu_Size: {} + {} / {}'.format(ACTIVE_SIZE,
#                                                                         len(needed_label_idx) - ACTIVE_SIZE,
#                                                                         total_label_num))
#         # use needed true_label for labeling
#         y_update_needed = np.array(y_update)[needed_label_idx]
#
#         my_X_update, my_y_update, keep_mask = \
#             Explain_and_Label_cluster_strutrue(X_update, y_update_needed,
#                                                needed_label_idx, sample_idxs_list,
#                                                Pred_bias, CLUSTER_improve_FLAG, y_update)
#     return my_X_update, my_y_update, update_flag, keep_mask
#
#
# def cluster4(test_x, test_y, anom_score,
#              label_num=200, unique_label=2, window_size=10000,
#              PCA_parameter=0.99, clf='KMeans', max_cluster_num=8):
#     '''伪标签算法，test_x, test_y, anom_score，输出test_x, acitive_&_clustered_pred_y'''
#     #######################################################
#     # 0.1define Active Size
#     # __test formula for Active Size__:  y = 0.1*label_num/100/(x + m) + b, limited by (x1,y1), (x2,y2)
#     #######################################################
#     my_x1, my_y1 = 0, window_size  # (x1, y1) = ( 0, 10000)
#     my_x2, my_y2 = unique_label, 0  # (x2, y2) = ( 2, 0)
#     m = np.sqrt(1. * np.abs(my_x1 - my_x2 - 1e-99) / np.abs(my_y1 - my_y2 - 1e-99) + 1) - 1
#     b = -1. / (m + 2 - 1e-99)
#
#     Active_size = 0.1 * label_num / 100 / (anom_score[0] - anom_score[20] + m) + b
#     print('Active_size: {}'.format(Active_size))
#     Active_size = max(min(Active_size, 0.3 * label_num), 0.1 * label_num)
#     print('[*] Active_size: {}'.format(Active_size))
#     Active_size = int(Active_size)
#     #######################################################
#     # 0.2 optional parameter
#     #######################################################
#     # PCA_parameter = 'mle', 0.99, 2
#     # max_cluster_num = 8, 10
#     # Active_size = int(label_num * 0.2), 40
#     # single_cluster_num = 20
#
#     #######################################################
#     # 1. Start Acive_learning turn
#     #######################################################
#     import copy
#
#     print('-' * 10 + 'Try fake labeling...' + '-' * 10)
#     split_test_y1 = np.array(test_y[:Active_size])
#     split_test_x2 = copy.copy(test_x[Active_size:])
#     split_test_y2 = np.array(test_y[Active_size:])
#     if len(split_test_x2) <= 0:
#         print('-' * 10 + 'Done labeling.' + '-' * 10)
#         return test_x, split_test_y1
#
#     #######################################################
#     # 2.1 Start try Clustering turn
#     #######################################################
#     set_random_seed(1)
#     bestK, best_sih = 0, 0
#     # PCA_formed_X = split_test_x2
#     clusters = np.arange(2, max_cluster_num)
#
#     from sklearn.decomposition import PCA
#     pca = PCA(n_components=PCA_parameter)
#     try:  # 此处可能出现数学域错误，需要修改pca.py文件
#         PCA_formed_X = pca.fit_transform(split_test_x2)
#     except:
#         split_test_x2 = np.array(split_test_x2) + 1e-99
#         PCA_formed_X = pca.fit_transform(split_test_x2)
#     # for j in parameter:
#     if True:
#         for i_cluster in clusters:
#             if clf.lower() == 'KMeans'.lower():
#                 from sklearn.cluster import KMeans
#                 model = KMeans(n_clusters=i_cluster, n_init=10)
#             elif clf.lower() == 'GaussianMixture'.lower():
#                 from sklearn.mixture import GaussianMixture
#                 model = GaussianMixture(n_components=i_cluster, covariance_type='full', random_state=0)
#             else:
#                 model = None
#             # 模型拟合
#             model.fit(PCA_formed_X)
#             # 为每个示例分配一个集群
#             PCA_formed_yhat = model.predict(PCA_formed_X)
#             sih = metrics.silhouette_score(PCA_formed_X, PCA_formed_yhat)
#             if sih >= best_sih:
#                 print('[-] new best at: {} :{:4d}, clusters_num :{:4d}, sih:{}'.format(
#                     PCA_parameter, len(PCA_formed_X[0]), i_cluster, sih))
#                 best_sih = sih
#                 bestK = i_cluster
#
#     #######################################################
#     # 2.2 Decide Clustering_model, Fit and Cluster
#     #######################################################
#     set_random_seed(2)
#     print('Applying labeling...')
#
#     if clf.lower() == 'KMeans'.lower():
#         from sklearn.cluster import KMeans
#         model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
#         # model = KMeans(n_clusters=bestK, n_init=10)
#     elif clf.lower() == 'GaussianMixture'.lower():
#         from sklearn.mixture import GaussianMixture
#         model = GaussianMixture(n_components=bestK, covariance_type='full', random_state=0)
#     else:
#         model = None
#     model.fit(PCA_formed_X)
#     # 为每个示例分配一个集群
#     PCA_formed_yhat = model.predict(PCA_formed_X)
#     # 检索唯一群集
#     clusters = np.unique(PCA_formed_yhat)
#
#     all_cluster_num = int(label_num - Active_size)
#     #######################################################
#     # 2.3 Start Sampling and Pred
#     #######################################################
#     set_random_seed(3)
#     way = 'MODE_2'
#
#     # 为每个聚类赋标签
#     if way == 'MODE_1':
#         single_cluster_num = all_cluster_num // len(clusters)
#         # __MODE 1__: balanced cluster_label
#         yy = np.array([0] * len(split_test_y2))
#         for i_cluster in clusters:
#             # 获取此群集的示例的行索引
#             row_ix = np.where(PCA_formed_yhat == i_cluster)
#             row = np.hstack(np.ravel(row_ix))
#             if len(row) <= single_cluster_num:
#                 yy[row] = split_test_y2[row]
#             else:
#                 n = np.random.choice(row, size=single_cluster_num, replace=False)
#
#                 yy[row] = 1 if sum(split_test_y2[n]) >= single_cluster_num // 2 else 0
#         print('-' * 10 + 'Done labeling.' + '-' * 10)
#         return test_x, np.append(split_test_y1, yy)
#
#     elif way == 'MODE_2':
#         # __MODE 2__: im-balanced cluster_label
#         row_ixs = []
#         total_len = len(test_y)
#         for i_cluster in clusters:
#             # 获取此群集的示例的行索引
#             row_ix = np.where(PCA_formed_yhat == i_cluster)[0]
#             if len(row_ix) * all_cluster_num <= total_len:
#                 # 当前组过小，但至少需分配一个标签
#                 total_len -= len(row_ix)
#                 all_cluster_num -= 1
#                 row_ixs.append(None)
#             else:
#                 row_ixs.append(row_ix)
#
#         yy = np.array([0] * len(split_test_y2))
#         for row_ix in row_ixs:
#             if row_ix is None: continue
#             # 只为每个个数足够多的类采样并分配标签
#             get_label_num = int(len(row_ix) * all_cluster_num / total_len)
#             temp_n = np.random.choice(np.arange(0, len(row_ix)), size=get_label_num, replace=False)
#
#             if sum(split_test_y2[row_ix[temp_n]]) >= get_label_num // 2:
#                 yy[row_ix] = 1
#             else:
#                 yy[row_ix] = 0
#         print('-' * 10 + 'Done labeling.' + '-' * 10)
#         return test_x, np.append(split_test_y1, yy)
#     else:
#         print('[***] Not know how to decide sampling.')
#         return -1
#
#
# def cluster2(test_x, test_y, human_labels):
#     '''伪标签算法，输入ndarray，输出ndarray'''
#     from sklearn.cluster import KMeans
#     from sklearn.decomposition import PCA
#     import copy
#     X = copy.copy(test_x)
#
#     print('-' * 10 + 'Try fake labeling...' + '-' * 10)
#     clusters = np.arange(2, 8)
#     parameter = np.arange(2, 10)
#     bestK = 0
#     best_sih = 0
#     best_pca = 0
#     tempX = X
#     for j in parameter:
#         # print(tempX.shape)
#         pca = PCA(n_components=j)
#         tempX = pca.fit_transform(tempX)
#         for i in clusters:
#             model = KMeans(n_clusters=i)
#             # 模型拟合
#             model.fit(tempX)
#             # 为每个示例分配一个集群
#             yhat = model.predict(tempX)
#             sih = metrics.silhouette_score(tempX, yhat)
#             if sih >= best_sih:
#                 print('[-] new best at: parameter:{:4d}, clusters:{:4d}, sih:{}'.format(j, i, sih))
#                 best_sih = sih
#                 bestK = i
#                 best_pca = j
#         tempX = X
#
#     print('Applying labeling...')
#     pca = PCA(n_components=best_pca)
#     tempX = pca.fit_transform(X)
#     model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
#     model.fit(tempX)
#     # 为每个示例分配一个集群
#     yhat = model.predict(tempX)
#     # 检索唯一群集
#     clusters = np.unique(yhat)
#     # 为每个聚类赋标签
#     row_ixs = []
#     total_len = len(test_y)
#     for cluster in clusters:
#         # 获取此群集的示例的行索引
#         row_ix = np.where(yhat == cluster)[0]
#         if len(row_ix) * human_labels <= total_len:
#             # 当前组过小，但至少需分配一个标签
#             total_len -= len(row_ix)
#             human_labels -= 1
#             row_ixs.append(None)
#         else:
#             row_ixs.append(row_ix)
#
#     yy = np.array([0] * len(test_y))
#     for row_ix in row_ixs:
#         if row_ix is None: continue
#         get_label_num = int(len(row_ix) * human_labels / total_len)
#         temp_n = np.random.choice(np.arange(0, len(row_ix)), size=get_label_num, replace=False)
#
#         if sum(test_y[row_ix[temp_n]]) >= get_label_num / 2:
#             yy[row_ix] = 1
#         else:
#             yy[row_ix] = 0
#     print('-' * 10 + 'Done labeling.' + '-' * 10)
#     return test_x, yy


if __name__ == '__main__':
    pass
    y_true = [3, 0]
    y_pred = [2, 3]
    evaluate_true_pred_label(y_true, y_pred, desc='')
