import torch
import torch.nn as nn
import torch.nn.functional as F

# 该代码主要功能是计算出注意力机制attention，根据attention计算出节点表示h_prime
class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        
# 用于计算节点的权重、自注意力权重和邻居注意力权重。
# 这些参数都是通过 nn.Parameter 包装后的张量，表示它们是模型可训练的参数
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

# 单个GAT的前向传播
# 输入是节点特征 input、邻接矩阵 adj、邻接矩阵的掩码 M
# 首先根据节点特征和权重矩阵 W 计算节点表示 h
# 然后通过计算自注意力机制（self-attention）得到注意力系数
# 接着根据注意力系数和邻接矩阵的掩码计算注意力矩阵 attn_dense
# 再通过 softmax 函数得到归一化的注意力矩阵 attention
# 最后，根据注意力矩阵和节点表示计算出聚合后的节点表示 h_prime
# 并根据 concat 参数决定是否返回经过激活函数处理后的结果

    def forward(self, input, adj, M, concat=True):
        
        # 输入特征与权重矩阵相乘，得到h
        h = torch.mm(input, self.W)    
        # attn_for_self 是节点对自身的注意力权重，attn_for_neighs 是节点对其邻居节点的注意力权重 
        # attn_for_dense计算了节点对自身和其邻居节点的注意力权重之和
        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1) 
        # 得到 cij
        attn_dense = torch.mul(attn_dense, M)   # +接近矩阵 M 用来屏蔽或减少对某些节点的注意力
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)  LeakyRelu激活函数：在输入小于零的部分引入一个小的线性斜率，以避免梯度消失问题
        

        zero_vec = -9e15 * torch.ones_like(adj) #为了进行 softmax 操作时，有效忽略零元素
        adj = torch.where(adj > 0, attn_dense, zero_vec) #将邻接矩阵中的非零元素替换为注意力权重，而将零元素替换为上面创建的负数张量
        attention = F.softmax(adj, dim=1) # softmax 操作，将注意力权重转换为概率分布，表示了每个节点对所有其他节点的注意力权重

        h_prime = torch.matmul(attention, h) # 经过注意力机制后的节点表示 h_prime
        
        # 是否将 h_prime 与原始节点表示 h 进行连接
        if concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )