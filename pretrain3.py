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
from model3 import GAT
from evaluation import eva


def pretrain(dataset):
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    # Adam优化器，用于训练模型参数
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    x = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    for epoch in range(args.max_epoch):
        model.train()
        A1_pred , A2_pred, z = model(x, adj, M)
        loss = F.binary_cross_entropy(A2_pred.view(-1), adj_label.view(-1))
        loss = loss + args.p * torch.norm(abs(A1_pred - A2_pred))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 梯度清零过程中进行k-means聚类
        with torch.no_grad():
            _,_, z = model(x, adj, M)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
        if epoch % 2 == 0:
            torch.save(
                model.state_dict(), f"./pretrain/predaegc_{args.name}_{epoch}.pkl"
            )


if __name__ == "__main__":
    # 从命令行接收参数并且解析为args对象的属性值
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="Citeseer")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
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
    # 获取数据集
    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == "Citeseer":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 6
        args.p = 0.02
        args.hidden_size = 256
        args.embedding_size = 24
    elif args.name == "Cora":
        args.lr = 0.02
        args.k = None
        args.p = 0.02
        args.n_clusters = 7
        args.hidden_size = 64
        args.embedding_size = 12
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.p = 0.02
        args.n_clusters = 3
      
        
    else:
        args.k = None

    args.input_dim = dataset.num_features

    print(args)
    pretrain(dataset)