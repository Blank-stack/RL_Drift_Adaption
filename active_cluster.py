import numpy as np
import pandas as pd

from my_tools import *


def generate_active_weight(anom_score, total_label_num, ACTIVE_WEIGHT_0=0.4):
    if len(anom_score) < total_label_num:
        return -1
    ACTIVE_SIZE_0 = int(ACTIVE_WEIGHT_0 * total_label_num)

    fig_X, fig_y = np.arange(ACTIVE_SIZE_0), anom_score[:ACTIVE_SIZE_0]
    fun_y = np.poly1d(np.polyfit(fig_X, fig_y, 1))  # 组合方程
    K_40 = np.abs(fun_y(1) - fun_y(0)) + 1e-99  # 拟合后y值

    fig_X, fig_y = np.arange(total_label_num), anom_score[:total_label_num]
    fun_y = np.poly1d(np.polyfit(fig_X, fig_y, 1))  # 组合方程
    K_200 = np.abs(fun_y(1) - fun_y(0)) + 1e-99  # 拟合后y值

    # ACTIVE_WEIGHT = 0.2 * (1 - np.exp(-K_0 / K_40)) / (1 - np.exp(-1)) * np.exp(-K_40)
    ACTIVE_WEIGHT = ACTIVE_WEIGHT_0 * (1 - np.exp(-K_200 / K_40)) / (1 - np.exp(-1)) * np.exp(-K_200)

    ACTIVE_WEIGHT = max(min(ACTIVE_WEIGHT, 1), 0.05)
    return ACTIVE_WEIGHT


def Generate_cluster_strutrue(test_x, anom_score, label_num=200, ACTIVE_WEIGHT=0.2,
                              PCA_parameter=0.99, clf='KMeans', max_cluster_num=8):
    #######################################################
    # 1. Start Acive_learning turn
    #######################################################
    # 1 主动学习参数
    ACTIVE_WEIGHT = max(min(ACTIVE_WEIGHT, 1), 0)
    Active_size = int(label_num * ACTIVE_WEIGHT)
    print('[*] Step 1.1: Active_size: {} / {}'.format(Active_size, label_num))

    Active_idxs = np.arange(min(len(test_x), Active_size))
    if label_num - Active_size <= 0:
        print('[*] Lack of samples, no need to cluster samples.')
        print('\t' + '-' * 10 + 'Done Active Turn.' + '-' * 10)
        return Active_idxs, [[Active_size, Active_idxs]]
        # return test_x, Active_idxs

    if len(test_x) <= label_num:
        total_idxs = np.arange(len(test_x))
        print('[*] Drift samples less than label_num, no need to cluster samples.')
        print('\t' + '-' * 10 + 'Done Mixed Turn.' + '-' * 10)
        return total_idxs, [[len(test_x), total_idxs]]

    #######################################################
    # 2.1 Start try Clustering turn
    #######################################################
    print('[*] Step 1.2: Cluster_size: {} / {}'.format(label_num - Active_size, label_num))
    print('-' * 10 + 'Try fake labeling...' + '-' * 10)
    import copy

    split_test_x2 = copy.copy(test_x[Active_size:])

    set_random_seed(1)
    bestK, best_sih = 0, 0
    Trying_Clusters_List = np.arange(2, max_cluster_num)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_parameter)
    try:  # 此处可能出现数学域错误，需要修改pca.py文件
        PCA_formed_X = pca.fit_transform(split_test_x2)
    except:
        split_test_x2 = np.array(split_test_x2) + 1e-99
        PCA_formed_X = pca.fit_transform(split_test_x2)
    print('[-] PCA_parameter: {} , need_feature_len: {:4d}'.format(PCA_parameter, len(PCA_formed_X[0])))

    for i_cluster in Trying_Clusters_List:
        if clf.lower() == 'KMeans'.lower():
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=i_cluster, n_init=10)
        elif clf.lower() == 'GaussianMixture'.lower() or clf.lower() == 'GMM'.lower():
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(n_components=i_cluster, covariance_type='full', random_state=0)
        else:
            model = None

        # 模型拟合
        model.fit(PCA_formed_X)

        # 为每个示例分配一个集群
        PCA_formed_yhat = model.predict(PCA_formed_X)
        sih = metrics.silhouette_score(PCA_formed_X, PCA_formed_yhat)

        if sih >= best_sih:
            print('[-] new best at: clusters_num :{:4d}, sih:{}'.format(i_cluster, sih))
            best_sih = sih
            bestK = i_cluster

    #######################################################
    # 2.2 Decide Clustering_model, Fit and Cluster
    #######################################################
    set_random_seed(2)
    print('Applying cluster_num...')

    if clf.lower() == 'KMeans'.lower():
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
        # or by the following
        # model = KMeans(n_clusters=bestK, n_init=10)
    elif clf.lower() == 'GaussianMixture'.lower() or clf.lower() == 'GMM'.lower():
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=bestK, covariance_type='full', random_state=0)
    else:
        model = None
    model.fit(PCA_formed_X)
    # 为每个示例分配一个集群
    PCA_formed_yhat = model.predict(PCA_formed_X)
    # 检索唯一群集
    clusters = np.unique(PCA_formed_yhat)

    all_cluster_label_num = int(label_num - Active_size)
    needed_label_idx = [Active_idxs]
    sample_idxs_list = [[Active_size, Active_idxs]]
    #######################################################
    # 2.3 Start Sampling and Pred
    #######################################################
    set_random_seed(3)
    way = 'MODE_2'

    if way == 'MODE_1':
        # __MODE 1__: balanced cluster_label
        single_cluster_label_num = all_cluster_label_num // len(clusters)
        for i_cluster in clusters:
            # 获取此群集的示例的行索引
            clustered_sample_idxs = np.where(PCA_formed_yhat == i_cluster)[0] + Active_size
            if len(clustered_sample_idxs) <= single_cluster_label_num:
                needed_label_idx.append(clustered_sample_idxs)

                sample_idxs_list.append([1, clustered_sample_idxs])
            else:
                random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)),
                                              size=single_cluster_label_num, replace=False)
                needed_label_idx.append(clustered_sample_idxs[random_idx])

                sample_idxs_list.append([single_cluster_label_num, clustered_sample_idxs])

    elif way == 'MODE_2':
        # __MODE 2__: im-balanced cluster_label
        total_len = len(split_test_x2)
        for i_cluster in clusters:
            # 获取此群集的示例的行索引
            clustered_sample_idxs = np.where(PCA_formed_yhat == i_cluster)[0] + Active_size
            if len(clustered_sample_idxs) * all_cluster_label_num <= total_len:
                # 当前组过小，但至少需分配一个标签
                total_len -= len(clustered_sample_idxs)
                all_cluster_label_num -= 1

                random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)), size=1, replace=False)
                needed_label_idx.append(clustered_sample_idxs[random_idx])

                sample_idxs_list.append([1, clustered_sample_idxs])
            else:
                needed_label_idx.append(None)
                sample_idxs_list.append([None, clustered_sample_idxs])

        for i_idx, [Num, clustered_sample_idxs] in enumerate(sample_idxs_list):
            if Num is not None: continue
            # Num未设定，则需要确定需要的标签数，以及需要的样本下标及对应聚类
            # 只为每个个数足够多的类采样并分配标签
            get_label_num = int(len(clustered_sample_idxs) * all_cluster_label_num / total_len)
            random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)),
                                          size=min(get_label_num, len(clustered_sample_idxs)), replace=False)
            needed_label_idx[i_idx] = clustered_sample_idxs[random_idx]
            sample_idxs_list[i_idx][0] = get_label_num
    else:
        print('[***] Not know how to decide sampling.')
        return -1

    print('\t' + '-' * 10 + 'Done Applying cluster_num.' + '-' * 10)
    needed_label_idx_ravel = []
    for t in needed_label_idx:
        needed_label_idx_ravel.extend(t)
    return np.array(needed_label_idx_ravel), sample_idxs_list


def Generate_cluster_strutrue2(test_x, anom_score, Active_size, MIN_CLUSTER_SIZE, MIN_SIH=0.5,
                               PCA_parameter=0.99, clf='KMeans', max_cluster_num=8):
    #######################################################
    # 1. Start Acive_learning turn
    #######################################################
    # 1 主动学习参数
    if len(test_x) <= Active_size + MIN_CLUSTER_SIZE:
        print('[*] Drift samples less than label_num, no need to cluster samples.')
        total_idxs = np.arange(len(test_x))
        print('\t' + '-' * 10 + 'Done Mixed Turn.' + '-' * 10)
        return total_idxs, [[len(test_x), total_idxs]]

    Active_idxs = np.arange(Active_size)
    #######################################################
    # 2.1 Start try Clustering turn
    #######################################################
    print('-' * 10 + 'Try fake labeling...' + '-' * 10)
    import copy

    split_test_x2 = copy.copy(test_x[Active_size:])

    set_random_seed(1)
    bestK, best_sih = 0, 0
    Trying_Clusters_List = np.arange(2, max_cluster_num)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=PCA_parameter)
    try:  # 此处可能出现数学域错误，需要修改pca.py文件
        PCA_formed_X = pca.fit_transform(split_test_x2)
    except:
        split_test_x2 = np.array(split_test_x2) + 1e-99
        PCA_formed_X = pca.fit_transform(split_test_x2)
    print('[-] PCA_parameter: {} , need_feature_len: {:4d}'.format(PCA_parameter, len(PCA_formed_X[0])))

    for i_cluster in Trying_Clusters_List:
        if clf.lower() == 'KMeans'.lower():
            from sklearn.cluster import KMeans

            model = KMeans(n_clusters=i_cluster, n_init=10)
        elif clf.lower() == 'GaussianMixture'.lower() or clf.lower() == 'GMM'.lower():
            from sklearn.mixture import GaussianMixture

            model = GaussianMixture(n_components=i_cluster, covariance_type='full', random_state=0)
        else:
            model = None

        # 模型拟合
        model.fit(PCA_formed_X)

        # 为每个示例分配一个集群
        PCA_formed_yhat = model.predict(PCA_formed_X)
        sih = metrics.silhouette_score(PCA_formed_X, PCA_formed_yhat)

        if sih >= best_sih:
            print('[-] new best at: clusters_num :{:4d}, sih:{}'.format(i_cluster, sih))
            best_sih = sih
            bestK = i_cluster

    if best_sih < MIN_SIH:
        print('cluster sih_result: {} <= {}, abandon clustering'.format(best_sih, MIN_SIH))
        return Active_idxs, [[Active_size, Active_idxs]]

    #######################################################
    # 2.2 Decide Clustering_model, Fit and Cluster
    #######################################################
    set_random_seed(2)
    print('Applying cluster_num...')

    if clf.lower() == 'KMeans'.lower():
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
        # or by the following
        # model = KMeans(n_clusters=bestK, n_init=10)
    elif clf.lower() == 'GaussianMixture'.lower() or clf.lower() == 'GMM'.lower():
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(n_components=bestK, covariance_type='full', random_state=0)
    else:
        model = None
    model.fit(PCA_formed_X)
    # 为每个示例分配一个集群
    PCA_formed_yhat = model.predict(PCA_formed_X)
    # 检索唯一群集
    clusters = np.unique(PCA_formed_yhat)

    all_cluster_label_num = int(MIN_CLUSTER_SIZE / best_sih)
    needed_label_idx = [Active_idxs]
    sample_idxs_list = [[Active_size, Active_idxs]]
    #######################################################
    # 2.3 Start Sampling and Pred
    #######################################################
    set_random_seed(3)
    way = 'MODE_2'

    if way == 'MODE_1':
        # __MODE 1__: balanced cluster_label
        single_cluster_label_num = all_cluster_label_num // len(clusters)
        for i_cluster in clusters:
            # 获取此群集的示例的行索引
            clustered_sample_idxs = np.where(PCA_formed_yhat == i_cluster)[0] + Active_size
            if len(clustered_sample_idxs) <= single_cluster_label_num:
                needed_label_idx.append(clustered_sample_idxs)

                sample_idxs_list.append([1, clustered_sample_idxs])
            else:
                random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)),
                                              size=single_cluster_label_num, replace=False)
                needed_label_idx.append(clustered_sample_idxs[random_idx])

                sample_idxs_list.append([single_cluster_label_num, clustered_sample_idxs])

    elif way == 'MODE_2':
        # __MODE 2__: im-balanced cluster_label
        total_len = len(split_test_x2)
        for i_cluster in clusters:
            # 获取此群集的示例的行索引
            clustered_sample_idxs = np.where(PCA_formed_yhat == i_cluster)[0] + Active_size
            if len(clustered_sample_idxs) * all_cluster_label_num <= total_len:
                # 当前组过小，但至少需分配一个标签
                total_len -= len(clustered_sample_idxs)
                all_cluster_label_num -= 1

                random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)), size=1, replace=False)
                needed_label_idx.append(clustered_sample_idxs[random_idx])

                sample_idxs_list.append([1, clustered_sample_idxs])
            else:
                needed_label_idx.append(None)
                sample_idxs_list.append([None, clustered_sample_idxs])

        for i_idx, [Num, clustered_sample_idxs] in enumerate(sample_idxs_list):
            if Num is not None: continue
            # Num未设定，则需要确定需要的标签数，以及需要的样本下标及对应聚类
            # 只为每个个数足够多的类采样并分配标签
            get_label_num = int(len(clustered_sample_idxs) * all_cluster_label_num / total_len)
            random_idx = np.random.choice(np.arange(0, len(clustered_sample_idxs)),
                                          size=min(get_label_num, len(clustered_sample_idxs)), replace=False)
            needed_label_idx[i_idx] = clustered_sample_idxs[random_idx]
            sample_idxs_list[i_idx][0] = get_label_num
    else:
        print('[***] Not know how to decide sampling.')
        return -1

    print('\t' + '-' * 10 + 'Done Applying cluster_num.' + '-' * 10)
    needed_label_idx_ravel = []
    for t in needed_label_idx:
        needed_label_idx_ravel.extend(t)
    return np.array(needed_label_idx_ravel), sample_idxs_list


def Explain_and_Label_cluster_strutrue(test_X, yy_list, needed_label_idx, sample_idxs_list,
                                       Pred_bias=0.5, CLUSTER_improve_FLAG=True,
                                       _true_y_test=None, reli_thres=0.6):
    print('\t' + '-' * 10 + 'Start Try_Labeling...' + '-' * 10)
    Begin_Label_idx = 0
    pred_y = np.array([-1] * len(test_X))
    keep_mask = np.array([False] * len(test_X))
    for idx, [Num, clustered_sample_idxs] in enumerate(sample_idxs_list):
        End_Label_idx = Begin_Label_idx + Num
        # 主动学习
        if idx == 0:
            temp_yy = yy_list[Begin_Label_idx:End_Label_idx]
            for i_idx in range(len(temp_yy)):
                needed_idx_at = Begin_Label_idx + i_idx
                pred_y[needed_label_idx[needed_idx_at]] = temp_yy[i_idx]
                keep_mask[needed_label_idx[needed_idx_at]] = True
        # 聚类猜测
        else:
            temp_yy = yy_list[Begin_Label_idx:End_Label_idx]
            Anom_Score = np.sum(temp_yy) / len(temp_yy)
            if Anom_Score < Pred_bias:
                reliablity = (Pred_bias - Anom_Score) / (Pred_bias + 1e-99)
            else:
                reliablity = (Anom_Score - Pred_bias) / (1 - Pred_bias + 1e-99)

            if not CLUSTER_improve_FLAG:
                reliablity = 1
            if reliablity <= reli_thres:
                print('[*] Found cluster_{} NOT reliable, give-up clustering...'.format(idx))

                # 真标签复用
                if CLUSTER_improve_FLAG:
                    for i_idx in range(len(temp_yy)):
                        needed_idx_at = Begin_Label_idx + i_idx
                        pred_y[needed_label_idx[needed_idx_at]] = temp_yy[i_idx]
                        keep_mask[needed_label_idx[needed_idx_at]] = True
            else:
                my_label = 1 if Anom_Score >= Pred_bias else 0
                # 聚类伪标签
                pred_y[clustered_sample_idxs] = my_label
                keep_mask[clustered_sample_idxs] = True

                # 真标签复用
                if CLUSTER_improve_FLAG:
                    revise_true_NUM = 0
                    for i_idx in range(len(temp_yy)):
                        if pred_y[needed_label_idx[Begin_Label_idx]] != temp_yy[i_idx]:
                            revise_true_NUM += 1
                        needed_idx_at = Begin_Label_idx + i_idx
                        pred_y[needed_label_idx[needed_idx_at]] = temp_yy[i_idx]
                        keep_mask[needed_label_idx[needed_idx_at]] = True
                    # if revise_true_NUM>0:
                    print('cluster {}: 真标签复位数为{}/ {}.'.format(idx, revise_true_NUM, Num))
        # 检测聚类效果
        if _true_y_test is not None:
            temp_keep = keep_mask[clustered_sample_idxs]
            temp_true_y_test = np.array(_true_y_test)[clustered_sample_idxs]
            temp_pred_y = np.array(pred_y)[clustered_sample_idxs]
            evaluate_true_pred_label(temp_true_y_test[temp_keep],
                                     temp_pred_y[temp_keep],
                                     'cluster {}:'.format(idx), 'weak')
        Begin_Label_idx = End_Label_idx

    print('-' * 10 + 'Done labeling.' + '-' * 10)
    return np.array(test_X)[keep_mask], np.array(pred_y)[keep_mask], keep_mask

# 固定聚类
# def Mixed_Acitive_Cluster(X_update, y_update, anom_score=None, pred_update=None,
#                           update_flag=False, total_label_num=200, slight_label_num=5, Pred_bias=0.5,
#                           ACTIVE_WEIGHT='auto', CLUSTER_improve_FLAG=True):
def Mixed_Acitive_Cluster(d, anom_score=None, pred_update=None,
                            update_flag=False, total_label_num=None, slight_label_num=5, Pred_bias=0.5,
                            ACTIVE_WEIGHT='auto', CLUSTER_improve_FLAG=True):
    X_update, y_update = d.iloc[:, :-1].values, d.iloc[:, -1].values
    X_update, y_update = np.array(X_update), np.array(y_update)
    my_X_update, my_y_update = [], []
    keep_mask = np.array([])

    if update_flag:

        # 0. Prepare Turn
        print('[*] Step 1: Mixed_Acitive_Cluster NUM: {}'.format(total_label_num))
        if type(ACTIVE_WEIGHT) == type('') and ACTIVE_WEIGHT.lower() == 'AUTO'.lower():
            # 静态设置
            ACTIVE_WEIGHT = 0

            # 动态决定
            # ACTIVE_WEIGHT = generate_active_weight(anom_score, total_label_num, ACTIVE_WEIGHT_0=0.2)

        ACTIVE_WEIGHT = max(min(ACTIVE_WEIGHT, 1), 0)

        # 1.Start Generate Cluster_Structure
        needed_label_idx, sample_idxs_list = \
            Generate_cluster_strutrue(X_update, anom_score, total_label_num, ACTIVE_WEIGHT)

        print('[*****] Confirm Mixed_Act_Clu_Size: {} + {} / {}'.format(0, total_label_num, total_label_num))

        # use needed true_label for labeling
        y_update_needed = y_update[needed_label_idx]

        my_X_update, my_y_update, keep_mask = \
            Explain_and_Label_cluster_strutrue(X_update, y_update_needed,
                                               needed_label_idx, sample_idxs_list,
                                               Pred_bias, CLUSTER_improve_FLAG, y_update)
        my_X_update = pd.DataFrame(my_X_update)

        my_y_update = pd.DataFrame(my_y_update, columns=[' Label'])

    data = pd.concat([my_X_update, my_y_update], axis=1)

    return data

# 动态聚类
# def Mixed_Acitive_Cluster2(X_update, y_update, anom_score=None,
#                            update_flag=False, total_label_num=200, Pred_bias=0.5,
#                            ACTIVE_WEIGHT='auto', MIN_CLUSTER_WEIGHT=0.2, MIN_SIH=0.5, CLUSTER_improve_FLAG=True):
def Mixed_Acitive_Cluster2(d, anom_score=None,
                            update_flag=False, total_label_num=200, Pred_bias=0.5,
                            ACTIVE_WEIGHT='auto', MIN_CLUSTER_WEIGHT=0.2, MIN_SIH=0.1, CLUSTER_improve_FLAG=True):
    X_update, y_update = d.iloc[:, :-1].values, d.iloc[:, -1].values
    X_update, y_update = np.array(X_update), np.array(y_update)
    my_X_update, my_y_update = [], []
    keep_mask = np.array([])

    if update_flag:

        # 0. Prepare Turn
        # print('[*] Step 1: Mixed_Acitive_Cluster MAX_NUM: {}'.format(total_label_num))
        if type(ACTIVE_WEIGHT) == type('') and ACTIVE_WEIGHT.lower() == 'AUTO'.lower():
            # 静态设置
            # ACTIVE_WEIGHT = 0.2

            # 动态决定
            ACTIVE_WEIGHT = generate_active_weight(anom_score, total_label_num, ACTIVE_WEIGHT_0=0.2)

        ACTIVE_SIZE = int(max(min(ACTIVE_WEIGHT, 1), 0.05) * total_label_num)

        MIN_CLUSTER_SIZE = int(max(min(MIN_CLUSTER_WEIGHT, 1), 0.05) * total_label_num)
        # MIN_CLUSTER_SIZE = 88
        MIN_SIH = max(min(MIN_SIH, 1), 0.4)
        # MIN_SIH = 1
        if ACTIVE_SIZE + MIN_CLUSTER_SIZE / MIN_SIH > total_label_num:
            # print('Pre_Set MAX_Label to big, reset MIN_CLUSTER_SIZE...')
            MIN_CLUSTER_SIZE = (total_label_num - ACTIVE_SIZE) * MIN_SIH

        # 1.Start Generate Cluster_Structure
        needed_label_idx, sample_idxs_list = \
            Generate_cluster_strutrue2(X_update, anom_score, ACTIVE_SIZE, MIN_CLUSTER_SIZE, MIN_SIH)

        print('[*****] Confirm Mixed_Act_Clu_Size: {} + {} / {}'.format(ACTIVE_SIZE,
                                                                        len(needed_label_idx) - ACTIVE_SIZE,
                                                                        total_label_num))
        # use needed true_label for labeling
        y_update_needed = y_update[needed_label_idx]

        my_X_update, my_y_update, keep_mask = \
            Explain_and_Label_cluster_strutrue(X_update, y_update_needed,
                                               needed_label_idx, sample_idxs_list,
                                               Pred_bias, CLUSTER_improve_FLAG, y_update)
    my_X_update = pd.DataFrame(my_X_update)
    # print(my_X_update)
    my_y_update = pd.DataFrame(my_y_update, columns=[' Label'])
    # print(my_y_update)
    data = pd.concat([my_X_update, my_y_update], axis=1)

    return data


def main():
    # ACTIVE_WEIGHT = 1
    # ACTIVE_WEIGHT = 0.2
    # ACTIVE_WEIGHT = 0
    ACTIVE_WEIGHT = generate_active_weight(anom_score, total_label_num, 0.2)
    anom_score=None

    # X_update4, y_update4, update_flag, keep_mask = Mixed_Acitive_Cluster(X_update, y_update, anom_score,
    #                                                                      pred_update, update_flag,
    #                                                                      total_label_num, 0,
    #                                                                      Pred_bias, ACTIVE_WEIGHT,
    #                                                                      CLUSTER_improve_FLAG)
    X_update4, y_update4, update_flag, keep_mask = Mixed_Acitive_Cluster2(X_update, y_update, anom_score,
                                                                          update_flag, total_label_num,
                                                                          Pred_bias, ACTIVE_WEIGHT, 0.2, 0.5,
                                                                          CLUSTER_improve_FLAG)
if __name__ == '__main__':
    main()
