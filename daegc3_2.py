import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from torch_geometric.datasets import Planetoid

import utils
from model3 import GAT
from evaluation import eva

# 自定义的神经网络都需要继承pythorch的nn.Moudle
class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        # 从args.pretrain_path加载权重和参数到CPU内存中，同时也加载到gat模型中
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer
        # 存储聚类中心的层（聚类中心嵌入），后续会进行根据训练更新
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size)) 
        # 初始化使数据变化保持稳定，并允许修改聚类中心层数据
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        A1_pred,A2_pred,z = self.gat(x, adj, M) # （隐式）调用gat模型的forward方法
        q = self.get_Q(z) #调用本模型（daegc）的get_Q方法

        return A1_pred, A2_pred,z, q
# Q分布计算
    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
# P分布计算
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def trainer(dataset):
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    with torch.no_grad():
        _,_, z = model.gat(data, adj, M)

    # get kmeans and pretrain cluster result
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    # 执行了K-means聚类算法，并将结果存储在 y_pred 
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    # 将K-means算法找到的簇中心点赋值给深度学习模型中的 cluster_layer
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device) #kmeans.cluster_centers_：K-means算法找到的簇中心点
    eva(y, y_pred, 'pretrain')
    
    
    max_acc = 0  # 初始化前一个性能评估值  

    for epoch in range(args.max_epoch):
        model.train()
        A1_pred,A2_pred, z, Q = model(data, adj, M)
        q = Q.detach().data.cpu().numpy().argmax(1)  
        acc, nmi, ari, f1 = eva(y, q, epoch)  # 计算当前的性能评估值
        if acc > max_acc:  # 如果当前性能评估值优于前一个值
            p = target_distribution(Q.detach())  # 更新 p
            max_acc = acc  # 更新前一个性能评估值
        kl_loss = F.kl_div(Q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A2_pred.view(-1), adj_label.view(-1))

        loss = 1000 * kl_loss + re_loss
       

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Citeseer')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == 'Citeseer':
      args.lr = 0.002
      args.k = None
      args.n_clusters = 6
      args.hidden_size = 256
      args.embedding_size = 24
    elif args.name == 'Cora':
      args.lr = 0.05
      args.k = None
      args.n_clusters = 7
      args.hidden_size = 64
      args.embedding_size = 12
    elif args.name == "Pubmed":
        args.lr = 0.05
        args.k = None
        args.n_clusters = 3
        args.hidden_size = 256
        args.embedding_size = 24
    else:
        args.k = None
    
    
    args.pretrain_path = f'./pretrain/predaegc_{args.name}_{args.epoch}.pkl'
    args.input_dim = dataset.num_features


    print(args)
    trainer(dataset)