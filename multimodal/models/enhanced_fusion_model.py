"""
增强版多模态融合模型模块

该模块提供更先进的多模态融合模型，包括：
1. 残差融合模型：引入残差连接，改善深层网络训练
2. 自编码器融合模型：通过自编码器进行特征提取和降维
3. 门控融合模型：使用门控机制动态调整特征重要性
4. 自适应学习融合模型：根据样本难度逐步学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from multimodal.models.fusion_model import MultiModalBase
from multimodal.models.residual_blocks import ResidualBlock, BottleneckBlock
from multimodal.models.se_blocks import SEBlock, CBAM


class AutoEncoder(nn.Module):
    """自编码器模块"""
    
    def __init__(self, input_size, latent_size, dropout_rate=0.3):
        """
        初始化自编码器
        
        参数:
            input_size (int): 输入特征维度
            latent_size (int): 潜在空间维度
            dropout_rate (float): Dropout比率
        """
        super(AutoEncoder, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size // 2, latent_size),
            nn.BatchNorm1d(latent_size),
            nn.ReLU()
        )
        
        # 解码器（用于预训练，推理时不使用）
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, input_size // 2),
            nn.BatchNorm1d(input_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size // 2, input_size)
        )
    
    def forward(self, x, return_decoder=False):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征
            return_decoder (bool): 是否返回解码器输出
            
        返回:
            torch.Tensor 或 tuple: 编码特征或(编码特征, 解码特征)
        """
        latent = self.encoder(x)
        
        if return_decoder:
            reconstructed = self.decoder(latent)
            return latent, reconstructed
        
        return latent


class ResidualFusionModel(MultiModalBase):
    """残差融合模型
    
    使用残差连接的深层神经网络进行特征融合。
    残差连接有助于解决深层网络的梯度消失问题，提高训练效率和模型性能。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, dropout_rate=0.3, num_classes=3):
        """
        初始化残差融合模型
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(ResidualFusionModel, self).__init__()
        
        # HiSeq特征提取
        self.hiseq_extractor = nn.Sequential(
            nn.Linear(hiseq_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # RPPA特征提取
        self.rppa_extractor = nn.Sequential(
            nn.Linear(rppa_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 残差块
        self.residual_blocks = nn.Sequential(
            ResidualBlock(hidden_size, hidden_size, dropout_rate),
            ResidualBlock(hidden_size, hidden_size, dropout_rate),
            ResidualBlock(hidden_size, hidden_size, dropout_rate)
        )
        
        # 注意力块
        self.attention = SEBlock(hidden_size, reduction=8)
        
        # 分类器
        self.classifier = nn.Linear(hidden_size, num_classes)
    
    def forward(self, hiseq_data, rppa_data):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            
        返回:
            torch.Tensor: 预测结果
        """
        # 特征提取
        hiseq_features = self.hiseq_extractor(hiseq_data)
        rppa_features = self.rppa_extractor(rppa_data)
        
        # 特征拼接
        combined = torch.cat([hiseq_features, rppa_features], dim=1)
        
        # 融合
        fused = self.fusion(combined)
        
        # 残差学习
        residual_out = self.residual_blocks(fused)
        
        # 注意力增强
        attended = self.attention(residual_out)
        
        # 分类
        output = self.classifier(attended)
        
        return output


class AutoEncoderFusionModel(MultiModalBase):
    """自编码器融合模型
    
    使用自编码器进行特征降维，然后融合提取的特征表示。
    自编码器有助于学习数据的潜在表示，提高模型对高维数据的处理能力。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, dropout_rate=0.3, num_classes=3):
        """
        初始化自编码器融合模型
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(AutoEncoderFusionModel, self).__init__()
        
        # 自编码器潜在空间大小
        hiseq_latent_size = min(hidden_size, hiseq_input_size // 4)
        rppa_latent_size = min(hidden_size // 2, rppa_input_size)
        
        # HiSeq自编码器
        self.hiseq_autoencoder = AutoEncoder(
            hiseq_input_size, 
            hiseq_latent_size, 
            dropout_rate
        )
        
        # RPPA自编码器
        self.rppa_autoencoder = AutoEncoder(
            rppa_input_size, 
            rppa_latent_size, 
            dropout_rate
        )
        
        # 融合层
        combined_size = hiseq_latent_size + rppa_latent_size
        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(hidden_size, hidden_size, dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
        # 重建损失权重
        self.reconstruction_weight = 0.1
    
    def forward(self, hiseq_data, rppa_data, return_reconstruction=False):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            return_reconstruction (bool): 是否返回重建结果
            
        返回:
            torch.Tensor 或 tuple: 预测结果或(预测结果, 重建结果)
        """
        if self.training and return_reconstruction:
            # 训练模式，返回重建结果
            hiseq_latent, hiseq_recon = self.hiseq_autoencoder(hiseq_data, return_decoder=True)
            rppa_latent, rppa_recon = self.rppa_autoencoder(rppa_data, return_decoder=True)
            
            # 特征拼接
            combined = torch.cat([hiseq_latent, rppa_latent], dim=1)
            
            # 融合
            fused = self.fusion_layers(combined)
            
            # 分类
            output = self.classifier(fused)
            
            return output, (hiseq_recon, rppa_recon)
        else:
            # 推理模式，只返回预测结果
            hiseq_latent = self.hiseq_autoencoder(hiseq_data)
            rppa_latent = self.rppa_autoencoder(rppa_data)
            
            # 特征拼接
            combined = torch.cat([hiseq_latent, rppa_latent], dim=1)
            
            # 融合
            fused = self.fusion_layers(combined)
            
            # 分类
            output = self.classifier(fused)
            
            return output
    
    def get_reconstruction_loss(self, hiseq_data, rppa_data):
        """
        计算重建损失
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            
        返回:
            torch.Tensor: 重建损失
        """
        _, hiseq_recon = self.hiseq_autoencoder(hiseq_data, return_decoder=True)
        _, rppa_recon = self.rppa_autoencoder(rppa_data, return_decoder=True)
        
        hiseq_loss = F.mse_loss(hiseq_recon, hiseq_data)
        rppa_loss = F.mse_loss(rppa_recon, rppa_data)
        
        return hiseq_loss + rppa_loss


class GatedAttention(nn.Module):
    """门控注意力机制"""
    
    def __init__(self, input_size, hidden_size):
        """
        初始化门控注意力
        
        参数:
            input_size (int): 输入特征维度
            hidden_size (int): 隐藏层大小
        """
        super(GatedAttention, self).__init__()
        
        self.linear = nn.Linear(input_size, hidden_size)
        self.gate = nn.Linear(input_size, hidden_size)
    
    def forward(self, x):
        """
        前向传播
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            torch.Tensor: 加权特征
        """
        features = self.linear(x)
        gates = torch.sigmoid(self.gate(x))
        
        return features * gates


class GatedFusionModel(MultiModalBase):
    """门控融合模型
    
    使用门控注意力机制动态调整特征重要性。
    门控机制允许模型动态控制信息流，根据输入自适应地强调相关特征，抑制无关特征。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, dropout_rate=0.3, num_classes=3):
        """
        初始化门控融合模型
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(GatedFusionModel, self).__init__()
        
        # HiSeq特征处理
        self.hiseq_encoder = nn.Sequential(
            nn.Linear(hiseq_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # RPPA特征处理
        self.rppa_encoder = nn.Sequential(
            nn.Linear(rppa_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 门控注意力
        self.gate_hiseq = GatedAttention(hidden_size, hidden_size)
        self.gate_rppa = GatedAttention(hidden_size, hidden_size)
        
        # 交叉注意力 - HiSeq注意RPPA
        self.cross_hiseq2rppa = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 交叉注意力 - RPPA注意HiSeq
        self.cross_rppa2hiseq = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            BottleneckBlock(hidden_size, hidden_size // 2, hidden_size, dropout_rate),
            
            CBAM(hidden_size),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
    
    def forward(self, hiseq_data, rppa_data):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            
        返回:
            torch.Tensor: 预测结果
        """
        # 编码
        hiseq_encoded = self.hiseq_encoder(hiseq_data)
        rppa_encoded = self.rppa_encoder(rppa_data)
        
        # 门控特征
        hiseq_gated = self.gate_hiseq(hiseq_encoded)
        rppa_gated = self.gate_rppa(rppa_encoded)
        
        # 交叉注意
        hiseq_rppa_cat = torch.cat([hiseq_gated, rppa_gated], dim=1)
        rppa_hiseq_cat = torch.cat([rppa_gated, hiseq_gated], dim=1)
        
        hiseq_weights = self.cross_rppa2hiseq(rppa_hiseq_cat)
        rppa_weights = self.cross_hiseq2rppa(hiseq_rppa_cat)
        
        hiseq_attended = hiseq_gated * hiseq_weights
        rppa_attended = rppa_gated * rppa_weights
        
        # 特征拼接
        combined = torch.cat([hiseq_attended, rppa_attended], dim=1)
        
        # 融合
        fused = self.fusion(combined)
        
        # 分类
        output = self.classifier(fused)
        
        return output


class SelfPacedFusionModel(MultiModalBase):
    """自适应学习融合模型
    
    根据样本的难度调整学习策略，先学习简单样本，逐步过渡到困难样本。
    这种方法有助于提高模型的泛化能力和鲁棒性。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, dropout_rate=0.3, num_classes=3):
        """
        初始化自适应学习融合模型
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(SelfPacedFusionModel, self).__init__()
        
        # 自编码器和注意力模块
        self.hiseq_autoencoder = AutoEncoder(hiseq_input_size, hidden_size // 2, dropout_rate)
        self.rppa_autoencoder = AutoEncoder(rppa_input_size, hidden_size // 4, dropout_rate)
        
        # 门控注意力
        self.hiseq_attention = SEBlock(hidden_size // 2, reduction=4)
        self.rppa_attention = SEBlock(hidden_size // 4, reduction=2)
        
        # 融合模块
        combined_size = (hidden_size // 2) + (hidden_size // 4)
        self.fusion = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            ResidualBlock(hidden_size, hidden_size, dropout_rate),
            ResidualBlock(hidden_size, hidden_size, dropout_rate),
            
            CBAM(hidden_size),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 样本难度评估网络
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(combined_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, hiseq_data, rppa_data, return_difficulty=False):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            return_difficulty (bool): 是否返回样本难度
            
        返回:
            torch.Tensor 或 tuple: 预测结果或(预测结果, 样本难度)
        """
        # 特征提取
        hiseq_features = self.hiseq_autoencoder(hiseq_data)
        rppa_features = self.rppa_autoencoder(rppa_data)
        
        # 注意力增强
        hiseq_attended = self.hiseq_attention(hiseq_features)
        rppa_attended = self.rppa_attention(rppa_features)
        
        # 特征拼接
        combined = torch.cat([hiseq_attended, rppa_attended], dim=1)
        
        # 样本难度评估
        difficulty = self.difficulty_estimator(combined)
        
        # 融合
        fused = self.fusion(combined)
        
        # 分类
        output = self.classifier(fused)
        
        if return_difficulty:
            return output, difficulty
        return output
    
    def get_spld_loss(self, outputs, targets, difficulty, threshold=0.5):
        """
        计算自适应学习损失
        
        参数:
            outputs (torch.Tensor): 模型输出
            targets (torch.Tensor): 目标标签
            difficulty (torch.Tensor): 样本难度
            threshold (float): 难度阈值
            
        返回:
            torch.Tensor: 加权损失
        """
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        
        # 基于难度加权损失
        weights = torch.where(difficulty < threshold, 
                             torch.ones_like(difficulty), 
                             (1 - difficulty) / (1 - threshold))
        
        # 应用权重
        weighted_loss = ce_loss * weights.squeeze()
        
        return weighted_loss.mean() 