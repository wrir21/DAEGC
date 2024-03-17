import numpy as np
from munkres import Munkres

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear

from sklearn import metrics

# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py
# cluster_acc 函数用于计算聚类任务的准确率（accuracy）和宏平均 F1 分数
def cluster_acc(y_true, y_pred):
    # 调整y_true的最小值，y_true从0开始
    y_true = y_true - np.min(y_true) 
    # 计算y_true和y_pred中的类别数量
    # 转换为set，除去重复的值  
    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    # 处理y_true和y_pred中的类别数量不匹配的问题
    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

# 计算了一系列分类评价指标——eva 函数适用于聚类任务的评估
# metrics.accuracy_score 函数计算了准确率（accuracy）
# metrics.f1_score 函数计算了宏平均 F1 分数（macro-average F1 score）
# metrics.precision_score 函数计算了宏平均精确率（macro-average precision）
# metrics.recall_score 函数计算了宏平均召回率（macro-average recall）
# metrics.f1_score 函数计算了微平均 F1 分数（micro-average F1 score）
# metrics.precision_score 函数计算了微平均精确率（micro-average precision）
# metrics.recall_score 函数计算了微平均召回率（micro-average recall）
    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro

# eva 函数用于评估聚类结果的性能，并打印出相应的评价指标
def eva(y_true, y_pred, epoch=0):  # 真实标签、预测标签、迭代次数
    acc, f1 = cluster_acc(y_true, y_pred) #真实标签、预测标签
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    print(f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
    return acc, nmi, ari, f1