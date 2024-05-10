import pickle

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.utils import shuffle
from tensorflow.keras.models import Model
import os
import numpy as np
import tensorflow.keras.backend as K
import pandas as pd
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from matplotlib import pyplot
from Explain import explainer

def penulti_output(x: np.ndarray, DQN: Model):
    inp = DQN.input
    penulti_func = K.function([inp], [DQN.layers[-2].output])  # 实现指定输入输出？决定网络处于训练或者测试
    latent_x = penulti_func(x)[0]

    return latent_x


# def writeResults(name, rocs, prs, train_times, test_times, file_path):
def writeResults(name, pres, recs, f1s, acc, train_times, test_times, file_path):
    # roc_mean = np.mean(rocs)
    # roc_std = np.std(rocs)
    # pr_mean = np.mean(prs)
    # pr_std = np.std(prs)
    pres_mean = np.mean(pres)
    recs_mean = np.mean(recs)
    f1s_mean = np.mean(f1s)
    pres_std = np.std(pres)
    recs_std = np.std(recs)
    f1s_std = np.std(f1s)
    acc_mean = np.mean(acc)
    acc_std = np.std(acc)

    train_mean = np.mean(train_times)
    train_std = np.std(train_times)
    test_mean = np.mean(test_times)
    test_std = np.std(test_times)

    header = True
    if not os.path.exists(file_path):
        header = False

    with open(file_path, 'a') as f:
        if not header:
            f.write("{}, {}, {}, {}, {}, {}, {}\n".format("Name",
                                                          "Precision(mean/std)",
                                                          "Recall(mean/std)",
                                                          "F1-score(mean/std)",
                                                          "Accuracy(mean/std)",
                                                          "Train time/s",
                                                          "Test time/s"))

        f.write("{}, {}/{}, {}/{}, {}/{}, {}/{}, {}/{}, {}/{}\n".format(name,
                                                                        pres_mean, pres_std,
                                                                        recs_mean, recs_std,
                                                                        f1s_mean, f1s_std,
                                                                        acc_mean, acc_std,
                                                                        train_mean, train_std,
                                                                        test_mean, test_std))


def calculate_ncm(train, cal, test, model):
    width = train.shape[1]
    train_x, train_y_true = train.iloc[:, :width - 1], train.iloc[:, width - 1]
    train_x = train_x.values
    cal_x, cal_y_true = cal.iloc[:, :width - 1], cal.iloc[:, width - 1]
    cal_x = cal_x.values

    cal_y_predict = model.predict_label(cal_x)
    train_b = model.predict_b(train_x)
    train_m = model.predict(train_x)
    cal_b = model.predict_b(cal_x)
    cal_m = model.predict(cal_x)

    test_X, test_y = test.iloc[:, :width - 1], test.iloc[:, width - 1]
    test_X = test_X.values
    test_y_predict = model.predict_label(test_X)
    test_b = model.predict_b(test_X)
    test_m = model.predict(test_X)

    train_y_true = train_y_true.values
    cal_y_true = cal_y_true.values
    test_y = test_y.values

    return train_b, train_m, train_y_true, cal_b, cal_m, cal_y_true, cal_y_predict, test_b, test_m, test_y, test_y_predict


def extend(old_train, old_cal, new_data):
    old_train = shuffle(old_train)
    old_cal = shuffle(old_cal)
    new_data = shuffle(new_data)
    l1 = len(old_train)
    l2 = len(old_cal)
    l3 = len(new_data)
    # old_train = old_train.iloc[:33000, :]
    old_train = old_train.iloc[:20000, :]
    # old_cal = old_cal.iloc[:17000, :]
    old_cal = old_cal.iloc[:10000, :]
    new_train = new_data.iloc[:int(l3 * 0.66), :]
    new_cal = new_data.iloc[int(l3 * 0.66):, :]
    # frame_train = [old_train, new_train]
    frame_train = [old_train, new_data]
    frame_cal = [old_cal, new_cal]
    train = pd.concat(frame_train)
    cal = pd.concat(frame_cal)
    train = shuffle(train)
    cal = shuffle(cal)

    return train, cal


def GMM(data):

    data = data.reset_index(drop=True)
    X = data.iloc[:, :data.shape[1] - 1].values
    # y = data.iloc[:, data.shape[1] - 1].values
    # 定义模型 GMM高斯模型 & K-means
    # model = GaussianMixture(n_components=2)
    model = KMeans(n_clusters=2)
    # 模型拟合
    model.fit(X)
    # 为每个示例分配一个集群
    yhat = model.predict(X)
    pred = pd.DataFrame(yhat)
    if int(pred.apply(lambda x: x.sum())) > 1 / 2 * len(pred):
        # pred中1多，取0
        data = data.loc[pred[pred[0] < 1].index]
    else:
        # pred中0多，取1
        data = data.loc[pred[pred[0] > 0].index]

    return data

def cluster(data):
    l = len(data)
    data = data.reset_index(drop=True)
    X = data.iloc[:, :data.shape[1] - 1].values
    y = data.iloc[:, data.shape[1] - 1].values
    clusters = np.arange(2, 8)
    parameter = np.arange(2, 10)
    bestK = 0
    best_sih = 0
    best_pca = 0
    tempX = X
    for j in parameter:
        # print(tempX.shape)
        pca = PCA(n_components=j)
        tempX = pca.fit_transform(tempX)
        for i in clusters:
            model = KMeans(n_clusters=i)
            # 模型拟合
            model.fit(tempX)
            # 为每个示例分配一个集群
            yhat = model.predict(tempX)
            # =========== 轮廓系数Silhouette ==============
            sih = silhouette_score(tempX, yhat)
            if sih >= best_sih:
                best_sih = sih
                bestK = i
                best_pca = j
        tempX = X
    pca = PCA(n_components=best_pca)
    tempX = pca.fit_transform(X)
    model = KMeans(n_clusters=bestK, init='k-means++', random_state=7)
    model.fit(tempX)
    # 为每个示例分配一个集群
    yhat = model.predict(tempX)
    # 检索唯一群集
    clusters = unique(yhat)
    # 为每个聚类赋标签
    for cluster in clusters:
        # 获取此群集的示例的行索引

        row_ix = where(yhat == cluster)
        row = np.hstack(np.ravel(row_ix))
        l0 = len(row_ix)
        N = 88 * l0 / l
        # n = np.random.choice(row, size=20, replace=True)
        n = np.random.choice(row, size=int(N), replace=True)
        # if sum(data.loc[n, ' Label']) >= 10:
        #     data.loc[row, ' Label'] = 1
        # else:
        #     data.loc[row, ' Label'] = 0
        # if sum(data.loc[n, ' Label']) >= 10:
        if sum(data.loc[n, ' Label']) >= int(N)/2:
            data.loc[row, ' Label'] = 1
        else:
            data.loc[row, ' Label'] = 0
    return data

def test_evaluate(x, y):
    try:
        num = x.shape[0]
    except:
        num = len(x)
    if num == 0:
        return
    else:
        print(num)

    print('-' * 10 + 'last' + '-' * 10)
    TP = sum([1 for i in range(num) if x[i] == y[i] == 1])
    print("TP:\t" + str(TP), end='\t|| ')
    FP = sum([1 for i in range(num) if x[i] == 1 and y[i] == 0])
    print("FP:\t" + str(FP), end='\t|| ')
    TN = sum([1 for i in range(num) if x[i] == y[i] == 0])
    print("TN:\t" + str(TN), end='\t|| ')
    FN = sum([1 for i in range(num) if x[i] == 0 and y[i] == 1])
    print("FN:\t" + str(FN))

    rec = (TP / (TP + FN)) if (TP + FN) != 0 else 0
    print("Recall: :\t" + str(rec), end='\t|| ')
    prec = (TP / (TP + FP)) if (TP + FP) != 0 else 0
    print("Precision: :\t" + str(prec))
    Accu = (TP + TN) / (TP + FP + TN + FN) if (TP + FP + TN + FN) != 0 else 0
    print("Accuracy: :\t" + str(Accu), end='\t|| ')

    F1 = 2 * rec * prec / (rec + prec) if (rec + prec) != 0 else 0
    print("F1: :\t" + str(F1))

    # Specity = TN / (TN + FP) if (TN + FP) != 0 else 0
    # print("Specity: :\t" + str(Specity), end='\t|| ')

    # G_mean = sqrt(Specity * prec)
    # print("G-mean: :\t" + str(G_mean))

    # save("result_overall_result.npy", [temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, temp_9, temp9])


def cache_data(model, data_path):
    model_folder_path = os.path.dirname(data_path)

    # if not os.path.exists(model_folder_path):
    #     os.makedirs(model_folder_path)

    print('Saving data to {}...'.format(data_path))
    with open(data_path, 'wb') as f:
        pickle.dump(model, f)
    print('Done cache_data.')

def f(x):

    Xmin = 0
    Xmax = 1
    a = 0.2
    b = 0.8
    y = a + (b - a) / (Xmax - Xmin) * (x - Xmin)

    return y

def select(data, model):

    L = len(data)
    A = data.loc[data[' Label'] == 1]
    N = data.loc[data[' Label'] == 0]
    L1 = len(A)

    A_X = A.iloc[:, :A.shape[1]-1].values
    N_X = N.iloc[:, :N.shape[1]-1].values
    siml_b = model.predict_b(N_X)
    siml_m = model.predict(A_X)
    NN = explainer(N, siml_m, N=30000*f(L1/L))
    AA = explainer(A, siml_b, N=30000*(1-f(L1/L)))
    NN = pd.DataFrame(NN, columns=data.columns)
    AA = pd.DataFrame(AA, columns=data.columns)
    result = pd.concat([NN,AA], axis=0)
    result = shuffle(result)
    # result = pd.DataFrame(np.concatenate([result.values]), columns=data.columns)

    return result
