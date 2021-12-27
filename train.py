import torch
from torch_geometric.data import Data
import torch.nn.functional as fun
import torch.nn as nn
from sklearn import metrics
import numpy as np
from model import ResGCN
from sklearn.model_selection import KFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, auc, precision_recall_curve, \
    matthews_corrcoef

import scipy.io as sio
import torch.backends.cudnn as cudnn


# 建议写成类、函数等模块化程序
# data = sio.loadmat('BRCA.mat')
# features = data['data'].T
# labels = data['targets']
# indexes = data['BRCA_indexes']
data = sio.loadmat('GBM.mat')
features = data['GBM_Gene_Expression'].T
labels = data['GBM_clinicalMatrix']
indexes = data['GBM_indexes']
# data = sio.loadmat('LUNG.mat')
# features = data['LUNG_Gene_Expression'].T
# labels = data['LUNG_clinicalMatrix']
# indexes = data['LUNG_indexes']

labels = labels.reshape(labels.shape[0])
path = "data/"
cites = path + "edges_gbm.csv"
# 索引字典，转换到从0开始编码
index_dict = dict()

edge_index = []
draw_edge_index = []

for i in range(indexes.shape[0]):
    index_dict[int(indexes[i])] = len(index_dict)
    print(index_dict)

with open(cites, "r") as f:
    edges = f.readlines()
    for edge in edges:
        start, end = edge.split(',')
        edge_index.append([index_dict[int(start)], index_dict[int(end)]])
        draw_edge_index.append([index_dict[int(start)], index_dict[int(end)]])


print(edge_index)
labels = torch.LongTensor(labels)
features = torch.FloatTensor(features)
edge_index = torch.LongTensor(edge_index).t()
print(edge_index)

# 训练
# 固定种子
seed =1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
torch.manual_seed(seed)
cudnn.benchmark = False
cudnn.deterministic = True

mask = torch.randperm(len(index_dict))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

cora = Data(x=features, edge_index=edge_index.contiguous(), y=labels).to(device)
print(cora)
print(features.shape[1])

p_mean = np.zeros(10)
r_mean = np.zeros(10)
f1score_mean = np.zeros(10)
ACC_mean = np.zeros(10)
ARS_mean = np.zeros(10)
MCC_mean = np.zeros(10)
AUC_mean = np.zeros(10)
PR_AUC_mean = np.zeros(10)
DBI_mean = np.zeros(10)
SS_mean = np.zeros(10)
k = 5
f_name = './data/results/'
for n in range(10):
    p = np.zeros(5)
    r = np.zeros(5)
    f1score = np.zeros(5)
    ACC = np.zeros(5)
    ARS = np.zeros(5)
    MCC = np.zeros(5)
    AUC = np.zeros(5)
    PR_AUC = np.zeros(5)
    DBI = np.zeros(5)
    SS = np.zeros(5)
    m = 0
    kfold = KFold(n_splits=k, shuffle=True, random_state=n*n+1)
    for train_mask, test_mask in kfold.split(mask):

        model = ResGCN(features.shape[1], 4).to(device)
        criterion = nn.CrossEntropyLoss().to(device)
        # criterion=FocalLoss(num_class=4,alpha=[0.275, 0.097, 0.413, 0.215]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
        # weight = torch.FloatTensor([0.28, 0.10, 0.40, 0.22]), size_average = True
        for epoch in range(50):
            optimizer.zero_grad()
            out = model(cora)
            loss = criterion(out[train_mask], cora.y[train_mask])
            # revise - change: loss -> loss.item()
            print('epoch: %d loss: %.4f' % (epoch, loss.item()))
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                model.eval()
                _, pred = model(cora).max(dim=1)
                correct = int(pred[train_mask].eq(cora.y[train_mask]).sum().item())
                print(correct)
                acc = correct / len(train_mask)
                print('Accuracy: {:.4f}'.format(acc))
                model.train()

            # 测试
        model.eval()
        out = model(cora)
        out=fun.softmax(out)
        X_test = out[test_mask]
        _, pred = X_test.max(dim=1)
        X_test = X_test.cuda().data.cpu().numpy()

        Y_test = pred
        Y_test = Y_test.cuda().data.cpu().numpy()

        targets_test = cora.y[test_mask]
        targets_test = targets_test.cuda().data.cpu().numpy()

        p[m] = precision_score(targets_test, Y_test, average='macro')
        r[m] = recall_score(targets_test, Y_test, average='macro')
        f1score[m] = f1_score(targets_test, Y_test, average='macro')
        ACC[m] = accuracy_score(targets_test, Y_test)
        ARS[m] = metrics.adjusted_rand_score(targets_test, Y_test)
        MCC[m] = matthews_corrcoef(targets_test, Y_test)
        DBI[m] = metrics.davies_bouldin_score(X_test, Y_test)
        SS[m] = metrics.silhouette_score(X_test, Y_test)

        y_one_hot = label_binarize(targets_test, np.arange(4))
        y_score = out[test_mask]
        y_score = y_score.cuda().data.cpu().numpy()
        fpr, tpr, thresholds_roc = metrics.roc_curve(y_one_hot.ravel(), y_score.ravel())
        AUC[m] = metrics.auc(fpr, tpr)
        pr, re, thresholds_pr = precision_recall_curve(y_one_hot.ravel(), y_score.ravel())
        PR_AUC[m] = auc(re, pr)


        name = f_name + 'each' + '_' + 'result_gbm' + '.txt'
        f = open(name, "a")
        print('第%d五倍交叉，交叉%d' % (n, m), file=f)
        print('Precision : ', p[m], file=f)
        print('Recall : ', r[m], file=f)
        print('f1score : ', f1score[m], file=f)
        print('ACC : ', ACC[m], file=f)
        print('ARI : ', ARS[m], file=f)
        print('MCC : ', MCC[m], file=f)
        print('AUC : ', AUC[m], file=f)
        print('PR_AUC : ', PR_AUC[m], file=f)
        print('silhouette_width : ', SS[m], file=f)
        print('DBI : ', DBI[m], file=f)
        f.close()
        m = m+1
    name = f_name + 'each' + '_' + 'result' + '_n_mean_gbm.txt'
    f = open(name, "a")
    print('第%d五倍交叉平均结果' % n, file=f)
    print('Precision : ', np.mean(p), file=f)
    print('Recall : ', np.mean(r), file=f)
    print('f1score : ', np.mean(f1score), file=f)
    print('ACC : ', np.mean(ACC), file=f)
    print('ARI : ', np.mean(ARS), file=f)
    print('MCC : ', np.mean(MCC), file=f)
    print('AUC : ', np.mean(AUC), file=f)
    print('PR_AUC : ', np.mean(PR_AUC), file=f)
    print('DBI : ', np.mean(DBI), file=f)
    print('silhouette_width : ', np.mean(SS), file=f)
    f.close()
    p_mean[n] = np.mean(p)
    r_mean[n] = np.mean(r)
    f1score_mean[n] = np.mean(f1score)
    ACC_mean[n] = np.mean(ACC)
    ARS_mean[n] = np.mean(ARS)
    MCC_mean[n] = np.mean(MCC)
    AUC_mean[n] = np.mean(AUC)
    PR_AUC_mean[n] = np.mean(PR_AUC)
    DBI_mean[n] = np.mean(DBI)
    SS_mean[n] = np.mean(SS)

s_p_mean = np.mean(p_mean)
s_r_mean = np.mean(r_mean)
s_f1score_mean = np.mean(f1score_mean)
s_ACC_mean = np.mean(ACC_mean)
s_ARS_mean = np.mean(ARS_mean)
s_MCC_mean = np.mean(MCC_mean)
s_AUC_mean = np.mean(AUC_mean)
s_PR_AUC_mean = np.mean(PR_AUC_mean)
s_DBI_mean = np.mean(DBI_mean)
s_SS_mean = np.mean(SS_mean)

name = f_name + 'each' + '_' + 'result_gbm' + '_mean.txt'
f = open(name, "a")
print('十次五倍交叉平均结果', file=f)
print('Precision : ', s_p_mean, file=f)
print('Recall : ', s_r_mean, file=f)
print('f1score : ', s_f1score_mean, file=f)
print('ACC : ', s_ACC_mean, file=f)
print('ARI : ', s_ARS_mean, file=f)
print('MCC : ', s_MCC_mean, file=f)
print('AUC : ', s_AUC_mean, file=f)
print('PR_AUC : ', s_PR_AUC_mean, file=f)
print('DBI : ', s_DBI_mean, file=f)
print('silhouette_width : ', s_SS_mean, file=f)
f.close()
