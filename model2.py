import torch
import torch.nn as nn
import torch.nn.functional as F

from layer import GATLayer

class GAT(nn.Module):
    # 堆叠了两个 GAT 层,每个 GAT 层都是一个独立的注意力层，用于学习输入数据的不同表示
    # 第一个 GAT 层将输入数据的特征映射到一个更高维度的隐藏空间   输入特征 -> 隐藏层
    # 第二个 GAT 层将隐藏空间的表示映射到一个更低维度的嵌入空间   隐藏层 -> 嵌入层
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

# 前向传播： 重构结构矩阵A
# 第一个 GAT 层 conv1:输入特征 x 和邻接矩阵 adj ——> 隐藏表示 h
# 第二个 GAT 层 conv2：h ——> 节点表示 z
# z 被归一化处理，且通过解码器 dot_product_decode,得到新的邻接矩阵 A_pred，归一化后的节点表示（张量）z

    def forward(self, x, adj, M):
        # 计算第一层GAT的输出
        h1 = self.conv1(x, adj, M)
        # 计算第二层GAT的输出
        h2 = self.conv2(h1, adj, M)
        #对第二层的GAT的输出进行归一化
        z = F.normalize(h2, p=2, dim=1)
        # 使用解码器计算第一层GAT的重构邻接矩阵 
        A1_pred = self.dot_product_decode(h1)   
        # 使用解码器计算重构邻接矩阵
        A2_pred = self.dot_product_decode(z)  # 使用A2_pred = self.dot_product_decode(h2)效果不好
        return A1_pred,A2_pred, z
    
    #解码器
    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred
    






