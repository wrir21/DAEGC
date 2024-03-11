import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

import utils
from model import GAT
from evaluation import eva

# 定义预训练函数：

# 创建一个GAT模型，并将其移到GPU/CPU上
def pretrain(dataset):
    # GAT模型：输入特征维度、隐藏层大小、嵌入维度、激活函数LeakyReLU的参数
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    # 定义优化器Adam：被优化的参数p、学习率lr、权重衰减参数weight_dency
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)



#数据预处理
    #数据集预处理，得到邻接矩阵 adj 和邻接矩阵标签、计算度矩阵M
    #将特征转换为Tensor 对象、将标签从 PyTorch Tensor 对象转换为 NumPy 数组
    dataset = utils.data_preprocessing(dataset) 
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)
    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()



#    模型在每个 epoch 中进行了一次完整的前向传播、损失计算、反向传播和参数更新的训练过程。
    for epoch in range(args.max_epoch):
        model.train()
        A_pred, z = model(x, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
        if epoch % 5 == 0:
            torch.save(
                model.state_dict(), f"./pretrain/predaegc_{args.name}_{epoch}.pkl"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="Citeseer")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == "Citeseer":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 6
    elif args.name == "Cora":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.input_dim = dataset.num_features

    print(args)
    pretrain(dataset)