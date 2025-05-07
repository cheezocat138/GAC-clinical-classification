"""
多模态融合模型模块

该模块提供用于整合HiSeq和RPPA数据的神经网络模型。
主要实现：
1. 基础多模态融合模型
2. 特征级融合（早期融合）模型
3. 决策级融合（晚期融合）模型
4. 注意力机制融合（混合融合）模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalBase(nn.Module):
    """多模态基础模型类"""
    
    def __init__(self):
        """初始化基础模型"""
        super(MultiModalBase, self).__init__()
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        raise NotImplementedError("子类必须实现forward方法")


class EarlyFusionModel(MultiModalBase):
    """特征级融合（早期融合）模型
    
    将两个模态的特征直接拼接在一起，输入到同一个神经网络中进行训练。
    这是最简单的融合方法，但忽略了不同模态的特性差异。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, dropout_rate=0.3, num_classes=3):
        """
        初始化早期融合模型
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(EarlyFusionModel, self).__init__()
        
        # 计算融合特征大小
        combined_input_size = hiseq_input_size + rppa_input_size
        
        # 定义神经网络层
        self.layers = nn.Sequential(
            nn.Linear(combined_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, num_classes)
        )
    
    def forward(self, hiseq_data, rppa_data):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            
        返回:
            torch.Tensor: 预测结果
        """
        # 特征拼接
        combined_features = torch.cat([hiseq_data, rppa_data], dim=1)
        
        # 通过神经网络
        return self.layers(combined_features)


class LateFusionModel(MultiModalBase):
    """决策级融合（晚期融合）模型
    
    为每个模态单独训练模型，在预测阶段合并各模型的输出。
    这种方法允许每个模态使用最适合的特征提取方法，但可能无法捕获模态间的交互关系。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, dropout_rate=0.3, num_classes=3):
        """
        初始化晚期融合模型
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(LateFusionModel, self).__init__()
        
        # HiSeq模型
        self.hiseq_model = nn.Sequential(
            nn.Linear(hiseq_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # RPPA模型
        self.rppa_model = nn.Sequential(
            nn.Linear(rppa_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        # 融合权重（可学习）
        self.fusion_weights = nn.Parameter(torch.ones(2) / 2)
    
    def forward(self, hiseq_data, rppa_data):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            
        返回:
            tuple: 融合结果, HiSeq结果, RPPA结果
        """
        # 各模态模型预测
        hiseq_output = self.hiseq_model(hiseq_data)
        rppa_output = self.rppa_model(rppa_data)
        
        # 归一化融合权重
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # 加权融合
        fusion_output = weights[0] * hiseq_output + weights[1] * rppa_output
        
        return fusion_output, hiseq_output, rppa_output


class AttentionFusionModel(MultiModalBase):
    """注意力机制融合模型（混合融合）
    
    使用注意力机制整合不同模态的特征，可以动态调整各模态的重要性。
    这种方法既保留了模态特异性，又能学习模态间的交互关系。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, dropout_rate=0.3, num_classes=3):
        """
        初始化注意力融合模型
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(AttentionFusionModel, self).__init__()
        
        # HiSeq特征提取器
        self.hiseq_encoder = nn.Sequential(
            nn.Linear(hiseq_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # RPPA特征提取器
        self.rppa_encoder = nn.Sequential(
            nn.Linear(rppa_input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 自注意力层 - 学习模态间的关系
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 2),
            nn.Softmax(dim=1)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, hiseq_data, rppa_data):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            
        返回:
            torch.Tensor: 预测结果
        """
        # 编码每个模态
        hiseq_features = self.hiseq_encoder(hiseq_data)
        rppa_features = self.rppa_encoder(rppa_data)
        
        # 拼接特征用于注意力计算
        combined_features = torch.cat([hiseq_features, rppa_features], dim=1)
        
        # 计算注意力权重
        attention_weights = self.attention(combined_features)
        
        # 加权融合特征
        weighted_hiseq = hiseq_features * attention_weights[:, 0].unsqueeze(1)
        weighted_rppa = rppa_features * attention_weights[:, 1].unsqueeze(1)
        
        # 拼接加权特征
        fused_features = torch.cat([weighted_hiseq, weighted_rppa], dim=1)
        
        # 分类层
        output = self.fusion(fused_features)
        
        return output


class CrossModalTransformerFusion(MultiModalBase):
    """跨模态Transformer融合模型
    
    使用Transformer架构进行模态间的交互和融合，可以捕获复杂的跨模态关系。
    这是一种更高级的融合方法，适合复杂的多模态学习任务。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, num_heads=4, 
                 num_layers=2, dropout_rate=0.3, num_classes=3):
        """
        初始化跨模态Transformer融合模型
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            num_heads (int): 注意力头数量
            num_layers (int): Transformer层数
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(CrossModalTransformerFusion, self).__init__()
        
        # 特征嵌入层
        self.hiseq_embedding = nn.Linear(hiseq_input_size, hidden_size)
        self.rppa_embedding = nn.Linear(rppa_input_size, hidden_size)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, 2, hidden_size))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, hiseq_data, rppa_data):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            
        返回:
            torch.Tensor: 预测结果
        """
        # 嵌入特征
        hiseq_embedded = self.hiseq_embedding(hiseq_data).unsqueeze(1)  # [batch_size, 1, hidden_size]
        rppa_embedded = self.rppa_embedding(rppa_data).unsqueeze(1)     # [batch_size, 1, hidden_size]
        
        # 拼接嵌入特征
        combined_embedded = torch.cat([hiseq_embedded, rppa_embedded], dim=1)  # [batch_size, 2, hidden_size]
        
        # 添加位置编码
        combined_embedded = combined_embedded + self.pos_encoder
        
        # Transformer编码
        transformer_output = self.transformer_encoder(combined_embedded)  # [batch_size, 2, hidden_size]
        
        # 平均池化或拼接
        flattened_output = transformer_output.reshape(transformer_output.size(0), -1)  # [batch_size, 2*hidden_size]
        
        # 分类
        output = self.classifier(flattened_output)
        
        return output


class HybridMultiModalNet(MultiModalBase):
    """混合多模态网络
    
    结合了多种融合策略，包括早期融合、晚期融合和注意力融合。
    通过学习不同融合策略的权重，自适应地选择最适合的融合方法。
    """
    
    def __init__(self, hiseq_input_size, rppa_input_size, hidden_size=256, dropout_rate=0.3, num_classes=3):
        """
        初始化混合多模态网络
        
        参数:
            hiseq_input_size (int): HiSeq特征维度
            rppa_input_size (int): RPPA特征维度
            hidden_size (int): 隐藏层大小
            dropout_rate (float): Dropout比率
            num_classes (int): 类别数量
        """
        super(HybridMultiModalNet, self).__init__()
        
        # 早期融合模型
        self.early_fusion = EarlyFusionModel(
            hiseq_input_size, rppa_input_size, hidden_size, dropout_rate, num_classes
        )
        
        # 晚期融合模型
        self.late_fusion = LateFusionModel(
            hiseq_input_size, rppa_input_size, hidden_size, dropout_rate, num_classes
        )
        
        # 注意力融合模型
        self.attention_fusion = AttentionFusionModel(
            hiseq_input_size, rppa_input_size, hidden_size, dropout_rate, num_classes
        )
        
        # 融合策略权重
        self.fusion_weights = nn.Parameter(torch.ones(3) / 3)
    
    def forward(self, hiseq_data, rppa_data):
        """
        前向传播
        
        参数:
            hiseq_data (torch.Tensor): HiSeq特征
            rppa_data (torch.Tensor): RPPA特征
            
        返回:
            torch.Tensor: 预测结果
        """
        # 早期融合
        early_output = self.early_fusion(hiseq_data, rppa_data)
        
        # 晚期融合
        late_output, _, _ = self.late_fusion(hiseq_data, rppa_data)
        
        # 注意力融合
        attention_output = self.attention_fusion(hiseq_data, rppa_data)
        
        # 归一化融合权重
        weights = F.softmax(self.fusion_weights, dim=0)
        
        # 加权融合
        fusion_output = (weights[0] * early_output + 
                         weights[1] * late_output + 
                         weights[2] * attention_output)
        
        return fusion_output


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建随机数据
    batch_size = 16
    hiseq_dim = 100
    rppa_dim = 50
    
    hiseq_data = torch.randn(batch_size, hiseq_dim)
    rppa_data = torch.randn(batch_size, rppa_dim)
    
    # 测试各种融合模型
    print("测试早期融合模型...")
    early_model = EarlyFusionModel(hiseq_dim, rppa_dim)
    early_output = early_model(hiseq_data, rppa_data)
    print(f"输出形状: {early_output.shape}")
    
    print("\n测试晚期融合模型...")
    late_model = LateFusionModel(hiseq_dim, rppa_dim)
    late_output, hiseq_out, rppa_out = late_model(hiseq_data, rppa_data)
    print(f"融合输出形状: {late_output.shape}")
    print(f"HiSeq输出形状: {hiseq_out.shape}")
    print(f"RPPA输出形状: {rppa_out.shape}")
    
    print("\n测试注意力融合模型...")
    attention_model = AttentionFusionModel(hiseq_dim, rppa_dim)
    attention_output = attention_model(hiseq_data, rppa_data)
    print(f"输出形状: {attention_output.shape}")
    
    print("\n测试跨模态Transformer融合模型...")
    transformer_model = CrossModalTransformerFusion(hiseq_dim, rppa_dim)
    transformer_output = transformer_model(hiseq_data, rppa_data)
    print(f"输出形状: {transformer_output.shape}")
    
    print("\n测试混合多模态网络...")
    hybrid_model = HybridMultiModalNet(hiseq_dim, rppa_dim)
    hybrid_output = hybrid_model(hiseq_data, rppa_data)
    print(f"输出形状: {hybrid_output.shape}")
    
    print("\n所有模型测试完成!") 