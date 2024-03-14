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
from model2 import GAT #导入model中的GAT类
from evaluation import eva


class DAEGC(nn.Module):
# DAEGC模型初始化

    # 定义聚类的数量、聚类中心的权重
    # 创建一个GAT模型实例gat，设置gat的属性，并且从指定路径（pkl文件）加载预训练的参数到gat中
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)



    # 前向传播：计算重构的邻接矩阵A_pred、节点表示z、q
    def forward(self, x, adj, M):
        A1_pred,A2_pred, z = self.gat(x, adj, M) 
        q = self.get_Q(z) 
        return A1_pred,A2_pred, z, q
  
    # 计算数据点和聚类中心之间的相似度，并返回相似度矩阵Q
    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q
    

# 基于给定的数据点与聚类中心的相似度矩阵 q，计算每个数据点在每个聚类中的权重
def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

# 训练模型
# 创建一个 DAEGC 模型的实例，定义了优化器，使用基于梯度下降的优化算法的Adam 优化器optimizer
def trainer(dataset):
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 


    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()
    with torch.no_grad():
        _, _,z = model.gat(data, adj, M)

    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pretrain')


    for epoch in range(args.max_epoch):
        model.train()
        # 按照原文逻辑更新P
        if epoch % args.update_interval == 0:
                # update_interval
                A1_pred,A2_pred, z, Q = model(data, adj, M)
                
                q = Q.detach().data.cpu().numpy().argmax(1)  # Q
                eva(y, q, epoch)

        A1_pred,A2_pred ,z, q = model(data, adj, M)
        p = target_distribution(Q.detach())
        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        # 计算重构损失 Lr    
        re_loss = F.binary_cross_entropy(A2_pred.view(-1), adj_label.view(-1))
        # 计算A1_pred和A2_pred的绝对差并添加到原始损失中
        abs_diff_loss = torch.sum(torch.abs(A1_pred - A2_pred))
        loss = 10 * kl_loss + re_loss
        # 计算总损失
        loss = loss + 0.001 * abs_diff_loss

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
      args.lr = 0.0001
      args.k = None
      args.n_clusters = 6
    elif args.name == 'Cora':
      args.lr = 0.0001
      args.k = None
      args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None
    
    
    args.pretrain_path = f'./pretrain/predaegc_{args.name}_{args.epoch}.pkl'
    args.input_dim = dataset.num_features


    print(args)
    trainer(dataset)