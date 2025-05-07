"""
残差块模块

该模块提供用于构建深度残差网络的各种残差块。
残差连接允许梯度直接流过网络，缓解深层网络的梯度消失问题。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """基本残差块
    
    包含两个卷积层和一个残差连接。
    如果输入和输出维度不同，则使用投影快捷连接。
    """
    
    def __init__(self, in_features, out_features, dropout_rate=0.3):
        """
        初始化残差块
        
        参数:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            dropout_rate (float): Dropout比率
        """
        super(ResidualBlock, self).__init__()
        
        # 主路径
        self.main_path = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # 快捷连接（如果输入/输出维度不同）
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            torch.Tensor: 输出特征
        """
        identity = self.shortcut(x)
        output = self.main_path(x)
        output += identity
        output = F.relu(output)
        
        return output


class BottleneckBlock(nn.Module):
    """瓶颈残差块
    
    使用1x1卷积降维和升维，中间是3x3卷积。
    这种结构可以减少参数量和计算量，同时保持性能。
    """
    
    def __init__(self, in_features, bottleneck_features, out_features, dropout_rate=0.3):
        """
        初始化瓶颈残差块
        
        参数:
            in_features (int): 输入特征维度
            bottleneck_features (int): 瓶颈层特征维度
            out_features (int): 输出特征维度
            dropout_rate (float): Dropout比率
        """
        super(BottleneckBlock, self).__init__()
        
        # 主路径
        self.main_path = nn.Sequential(
            # 降维
            nn.Linear(in_features, bottleneck_features),
            nn.BatchNorm1d(bottleneck_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 处理
            nn.Linear(bottleneck_features, bottleneck_features),
            nn.BatchNorm1d(bottleneck_features),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # 升维
            nn.Linear(bottleneck_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        
        # 快捷连接（如果输入/输出维度不同）
        self.shortcut = nn.Identity()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            torch.Tensor: 输出特征
        """
        identity = self.shortcut(x)
        output = self.main_path(x)
        output += identity
        output = F.relu(output)
        
        return output


class ResidualStack(nn.Module):
    """残差块堆叠
    
    将多个残差块堆叠在一起，形成一个深层残差网络。
    """
    
    def __init__(self, in_features, out_features, num_blocks=2, dropout_rate=0.3):
        """
        初始化残差块堆叠
        
        参数:
            in_features (int): 输入特征维度
            out_features (int): 输出特征维度
            num_blocks (int): 残差块数量
            dropout_rate (float): Dropout比率
        """
        super(ResidualStack, self).__init__()
        
        layers = []
        
        # 第一个残差块可能需要改变维度
        layers.append(ResidualBlock(in_features, out_features, dropout_rate))
        
        # 添加剩余的残差块
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_features, out_features, dropout_rate))
        
        self.stack = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            torch.Tensor: 输出特征
        """
        return self.stack(x)


class DenseBlock(nn.Module):
    """密集连接块
    
    类似于DenseNet的结构，每个层的输出都连接到后续所有层。
    这种结构可以更好地传播梯度，提高特征复用率。
    """
    
    def __init__(self, in_features, growth_rate, num_layers, dropout_rate=0.3):
        """
        初始化密集连接块
        
        参数:
            in_features (int): 输入特征维度
            growth_rate (int): 每一层输出的特征数量
            num_layers (int): 层数
            dropout_rate (float): Dropout比率
        """
        super(DenseBlock, self).__init__()
        
        self.layers = nn.ModuleList()
        current_features = in_features
        
        for i in range(num_layers):
            layer = nn.Sequential(
                nn.Linear(current_features, growth_rate),
                nn.BatchNorm1d(growth_rate),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            )
            self.layers.append(layer)
            current_features += growth_rate
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            torch.Tensor: 输出特征
        """
        features = [x]
        
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        
        return torch.cat(features, dim=1) 