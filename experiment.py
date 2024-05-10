import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import tensorflow as tf

tf.device("/cpu:0")
from DPLAN import DPLAN
from ADEnv import ADEnv
from utils import calculate_ncm, extend, GMM, cluster, test_evaluate, f
from utils import writeResults
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score
import half_transcend_ce.half_ce_siml as aaa
import warnings
from active_cluster import Mixed_Acitive_Cluster2, Mixed_Acitive_Cluster

warnings.filterwarnings("ignore")

runs = 1
model_path = "./model"
result_path = "./results"
result_file = "results.csv"

data_name = "x"
p = 10000
# Anomaly Detection Environment Settings
prob_au = 0.5
label_normal = 0
label_anomaly = 1
# DPLAN Settings
settings = {}
settings["hidden_layer"] = 20  # l
settings["memory_size"] = 200000  # M
# settings["warmup_steps"] = 10000  # 1w
settings["warmup_steps"] = 2000  # 1w
# settings["episodes"] = 10
settings["episodes"] = 2
settings["steps_per_episode"] = 2000  # 2000
settings["epsilon_max"] = 1
settings["epsilon_min"] = 0.1
# settings["epsilon_course"] = 10000  # 1w
settings["epsilon_course"] = 2000  # 1w
settings["minibatch_size"] = 32
settings["discount_factor"] = 0.99  # gamma
settings["learning_rate"] = 0.00025
settings["minsquared_gradient"] = 0.01
settings["gradient_momentum"] = 0.95
settings["penulti_update"] = 2000  # N
# settings["target_update"] = 10000  # K 1w
settings["target_update"] = 2000  # K 1w
if not os.path.exists(model_path):
    os.mkdir(model_path)
if not os.path.exists(result_path):
    os.mkdir(result_path)

# load data
source = pd.read_csv("./data/StreamCIC_.csv", index_col=None)
# source = pd.read_csv("./data/CICIDS2017/CICIDS2017.csv", index_col=None)
# source = pd.read_csv("./data/kitsune_RWDIDS_standard.csv", index_col=None)
# source.iloc[:, :100] = StandardScaler().fit_transform(source.iloc[:, :100])
# print(source)
# source.drop(columns=[' timestamp'], inplace=True)
# source = pd.read_csv("./data/rwdids/iP2V_RWDIDS_only.csv", index_col=None)
# source = pd.read_csv("./data/rwdids/recurrent_attack.csv", index_col=None)
length = len(source)
width = source.shape[1]
Train_Num = 50000
dataset_train = source.iloc[:Train_Num, :]
# dataset_test = source.iloc[Train_Num+1:, :]
dataset_test = source.iloc[350000:, :]
# dataset_test = source.iloc[293856:790635, :]
# golden_1 = dataset_train.loc[dataset_train[' Label'] == 1]
# golden_0 = dataset_train.loc[dataset_train[' Label'] == 0]
# d = pd.concat([golden_0, golden_1], axis=0)

np.random.seed(42)
tf.random.set_seed(42)
print()
pres = []
recs = []
f1s = []
accs = []
train_times = []
test_times = []
weights_file = os.path.join(model_path, "{}_weights.h5f".format(data_name))
tf.compat.v1.reset_default_graph()

# train
# A = len(dataset_train.loc[dataset_train[' Label']==1])
# N = len(dataset_train.loc[dataset_train[' Label']==0])
# bias = f(A /(A + N))

train_length = len(dataset_train)
dataset_train = shuffle(dataset_train)
train_train = dataset_train.iloc[:int(train_length * 0.66), :]
train_cal = dataset_train.iloc[int(train_length * 0.66):, :]  # 验证集
# training = train_train.values
training = dataset_train.values
env = ADEnv(dataset=training,
            # sampling_Du=size_sampling_Du,
            prob_au=prob_au,
            label_normal=label_normal,
            label_anomaly=label_anomaly,
            name=data_name)
# env = ADEnv(dataset=training,
#             # sampling_Du=size_sampling_Du,
#             prob_au=prob_au,
#             label_normal=label_normal,
#             label_anomaly=label_anomaly,
#             bias=bias,
#             name=data_name)
model = DPLAN(env=env,
              settings=settings)
# train_time = 0
train_start = time.time()
model.fit(weights_file=weights_file)
train_end = time.time()
train_time = train_end - train_start
print("Train time: {}/s".format(train_time))
# print('size of RL model:{}'.format(sys.getsizeof(model)))

# test the agent
# test_time = 0
test_start = time.time()
# prediction_set = []
i = 0
while len(dataset_test) > p:
    windows = dataset_test.iloc[:p, :]
    # windows = windows.values
    model.load_weights(weights_file)
    # ncm
    # detect time
    # detect_start = time.time()

    train_b, train_m, train_y_true, cal_b, cal_m, cal_y_true, cal_y_predict, test_b, test_m, test_y, test_y_predict = calculate_ncm(
        train_train, train_cal, windows, model)
    mask, rate, order_idx, anom_score = aaa.start(train_b, train_m, train_y_true,
                           cal_b, cal_m, cal_y_true, cal_y_predict,
                           test_b, test_m, test_y, test_y_predict)
    # prediction_set.append(test_y_predict)
    # pre = precision_score(test_y, test_y_predict)
    # rec = recall_score(test_y, test_y_predict)
    # f1 = f1_score(test_y, test_y_predict)
    # acc = accuracy_score(test_y, test_y_predict)
    # pres.append(pre)
    # recs.append(rec)
    # f1s.append(f1)
    # accs.append(acc)
    print(i)
    # keep\reject
    # reject1, reject0, keep1, keep0 = aaa.reject(windows, mask, test_y_predict, i)
    reject, keep = aaa.reject(windows, mask, test_y_predict)
    # ============== reject 排序 ======================
    reject = reject.values
    reject = [reject[i] for i in order_idx]
    reject = pd.DataFrame(reject)
    anom_score = [anom_score[i] for i in order_idx]
    # ============== reject 排序 ======================
    # detect_end = time.time()
    # detect_time = detect_end - detect_start
    # print("Detect time: {}/s".format(detect_time))
    # golden_0 = golden_0.sample(frac=0.5, axis=0)
    # golden_1 = golden_1.sample(frac=0.5, axis=0)

    if rate <= 0.2:

        # d = pd.concat([keep0, keep1])

        train_train, train_cal = extend(train_train, train_cal, keep)

        # train = d.sample(30000, axis=0)
        # train = pd.concat([train, keep0, keep1], axis=0)
        # train = train.sample(30000, axis=0)
        # golden_0 = pd.concat([golden_0, keep0], axis=0)
        # golden_1 = pd.concat([golden_1, keep1], axis=0)
        # d = pd.concat([golden_0, golden_1], axis=0)
        # train = shuffle(train)
        # train_length = len(train)
        # train_train = train.iloc[:int(train_length * 0.66), :]
        # train_cal = train.iloc[int(train_length * 0.66):, :]
        # golden_0 = pd.concat([golden_0, keep0], axis=0)
        # golden_1 = pd.concat([golden_1, keep1], axis=0)
        # d = pd.concat([golden_0, golden_1], axis=0)

    if rate >= 0.20:  # update
    # else:  # update

        # train = d.sample(30000, axis=0)
        # train = pd.concat([train, keep0, keep1], axis=0)
        # train = train.sample(30000, axis=0)
        # train = pd.concat([train, reject1, reject0], axis=0)
        # golden_0 = pd.concat([golden_0, keep0, reject0], axis=0)
        # golden_1 = pd.concat([golden_1, keep1, reject1], axis=0)
        # d = pd.concat([golden_0, golden_1], axis=0)
        # train = shuffle(train)
        # train_length = len(train)
        # train_train = train.iloc[:int(train_length * 0.66), :]
        # train_cal = train.iloc[int(train_length * 0.66):, :]


        # label_start = time.time()
        # ================================聚类打伪标签=====================================
        # reject = cluster(reject)

        # ================================动态打伪标签====================================
        reject = Mixed_Acitive_Cluster2(reject, anom_score, update_flag=True)

        # ================================单聚类/主动====================================
        # reject = Mixed_Acitive_Cluster(reject, anom_score, update_flag=True, total_label_num=100, ACTIVE_WEIGHT=0, CLUSTER_improve_FLAG=False)
        # ===================================================================================
        # label_end = time.time()
        # label_time = label_end - label_start
        # print("Label time: {}/s".format(label_time))

        # adapt_start = time.time()
        # ============= 老聚类 ==============
        # d = pd.concat([keep, reject])
        # ============= 保留/不保留keep ==============
        d = pd.DataFrame(np.concatenate([keep.values, reject.values]), columns=keep.columns)
        # d = pd.DataFrame(np.concatenate([reject.values]), columns=keep.columns)
        # ============= XX ==============
        # A = len(d.loc[d[' Label'] == 1])
        # N = len(d.loc[d[' Label'] == 0])
        # bias = f(A / (A + N))

        train_train, train_cal = extend(train_train, train_cal, d)
        #
        training = train_train.values
        # training = train.values
        env1 = ADEnv(dataset=training,
                     prob_au=prob_au,
                     label_normal=label_normal,
                     label_anomaly=label_anomaly,
                     name=data_name)
        # env1 = ADEnv(dataset=training,
        #              prob_au=prob_au,
        #              label_normal=label_normal,
        #              label_anomaly=label_anomaly,
        #              bias=bias,
        #              name=data_name)

        model.set_processor(env1)
        model.load_weights(weights_file)
        model.fit(env=env1, weights_file=weights_file)

        # adapt_end = time.time()
        # adapt_time = adapt_end - adapt_start
        # print("Adapt time: {}/s".format(adapt_time))

    dataset_test = dataset_test.iloc[p:, :]
    i = i + 1

test_X, test_y = dataset_test.iloc[:, :width - 1], dataset_test.iloc[:, width - 1]
test_X = test_X.values
test_y = test_y.values


model.load_weights(weights_file)

pred_y = model.predict_label(test_X)
test_evaluate(pred_y, test_y)
# prediction_set.append(pred_y)
# prediction_set = pd.DataFrame(prediction_set)
# prediction_set.to_csv("prediction_set.csv", index=False)
# pre = precision_score(test_y, pred_y)
# rec = recall_score(test_y, pred_y)
# f1 = f1_score(test_y, pred_y)
# acc = accuracy_score(test_y, pred_y)
#
# test_end = time.time()
# test_time = test_end - test_start
# print(i)
# print("Test time: {}/s".format(test_time))
#
# pres.append(pre)
# recs.append(rec)
# f1s.append(f1)
# accs.append(acc)
# train_times.append(train_time)
# test_times.append(test_time)
# print()
