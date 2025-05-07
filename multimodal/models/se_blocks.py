"""
Squeeze-and-Excitation模块

该模块提供不同类型的注意力机制，用于增强神经网络的特征表示能力。
主要实现：
1. SEBlock: 通道注意力机制，通过压缩和激励操作重新校准通道特征
2. CBAM: 卷积块注意力模块，结合通道和空间注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation块
    
    通过压缩和激励操作实现通道注意力机制。
    参考论文："Squeeze-and-Excitation Networks" (CVPR 2018)
    """
    
    def __init__(self, channel, reduction=16):
        """
        初始化SE块
        
        参数:
            channel (int): 输入通道数
            reduction (int): 压缩比例
        """
        super(SEBlock, self).__init__()
        
        # 压缩和激励层
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征 [batch_size, channel]
            
        返回:
            torch.Tensor: 加权特征 [batch_size, channel]
        """
        b, c = x.size()
        
        # 全局平均池化
        y = self.avg_pool(x.unsqueeze(-1)).view(b, c)
        
        # 压缩和激励
        y = self.fc(y).view(b, c)
        
        # 通道加权
        return x * y


class ChannelAttention(nn.Module):
    """通道注意力
    
    CBAM的通道注意力部分，同时使用平均池化和最大池化。
    """
    
    def __init__(self, in_planes, ratio=16):
        """
        初始化通道注意力
        
        参数:
            in_planes (int): 输入通道数
            ratio (int): 压缩比例
        """
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        self.shared_mlp = nn.Sequential(
            nn.Linear(in_planes, in_planes // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_planes // ratio, in_planes, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征 [batch_size, channel]
            
        返回:
            torch.Tensor: 通道注意力权重 [batch_size, channel]
        """
        b, c = x.size()
        x_unsqueezed = x.unsqueeze(-1)
        
        avg_out = self.shared_mlp(self.avg_pool(x_unsqueezed).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x_unsqueezed).view(b, c))
        
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力
    
    CBAM的空间注意力部分，同时使用平均池化和最大池化的结果生成空间注意力图。
    对于全连接网络，我们模拟这种效果。
    """
    
    def __init__(self, kernel_size=7):
        """
        初始化空间注意力
        
        参数:
            kernel_size (int): 卷积核大小
        """
        super(SpatialAttention, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(2, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征 [batch_size, channel]
            
        返回:
            torch.Tensor: 加权特征 [batch_size, channel]
        """
        b, c = x.size()
        
        # 计算统计信息作为空间特征
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # 拼接平均值和最大值
        stat = torch.cat([avg_out, max_out], dim=1)
        
        # 生成空间权重
        weight = self.mlp(stat)
        
        # 扩展权重维度进行广播
        spatial_weight = weight.expand_as(x)
        
        return x * spatial_weight


class CBAM(nn.Module):
    """卷积块注意力模块 (CBAM)
    
    结合通道和空间注意力机制的注意力模块。
    参考论文："CBAM: Convolutional Block Attention Module" (ECCV 2018)
    """
    
    def __init__(self, channel, ratio=16, kernel_size=7):
        """
        初始化CBAM
        
        参数:
            channel (int): 输入通道数
            ratio (int): 通道注意力的压缩比例
            kernel_size (int): 空间注意力的卷积核大小
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(channel, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征 [batch_size, channel]
            
        返回:
            torch.Tensor: 通过CBAM加权的特征 [batch_size, channel]
        """
        # 通道注意力
        x = x * self.channel_attention(x)
        
        # 空间注意力
        x = self.spatial_attention(x)
        
        return x


class GCNet(nn.Module):
    """全局上下文网络 (GCNet)
    
    使用全局上下文块进行特征增强。
    参考论文："GCNet: Non-local Networks Meet Squeeze-Excitation Networks and Beyond" (ICCVW 2019)
    """
    
    def __init__(self, in_channels, reduction=16):
        """
        初始化GCNet
        
        参数:
            in_channels (int): 输入通道数
            reduction (int): 压缩比例
        """
        super(GCNet, self).__init__()
        
        # 全局上下文建模
        self.context_modeling = nn.Sequential(
            nn.Linear(in_channels, 1, bias=False),
            nn.Softmax(dim=1)
        )
        
        # 转换
        self.transform = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.LayerNorm(in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.LayerNorm(in_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征 [batch_size, channel]
            
        返回:
            torch.Tensor: 加权特征 [batch_size, channel]
        """
        # 全局上下文建模
        context = self.context_modeling(x)
        
        # 计算全局上下文特征
        context_feature = torch.matmul(context.unsqueeze(1), x).squeeze(1)
        
        # 转换
        transformed = self.transform(context_feature)
        
        # 残差连接
        return x + transformed * x 