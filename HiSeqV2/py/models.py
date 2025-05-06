import torch
import torch.nn as nn
import torch.nn.functional as F
from config import TRAINING_PARAMS

# 定义Squeeze-and-Excitation模块
class SEBlock(nn.Module):
    """Squeeze-and-Excitation块，为模型添加通道注意力机制"""
    def __init__(self, channel, reduction=4):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x.unsqueeze(-1)).squeeze(-1)
        y = self.fc(y)
        return x * y.expand_as(x)

# 定义残差模块
class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, use_se=True):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        
        # SE注意力模块
        self.use_se = use_se
        if self.use_se:
            self.se = SEBlock(out_features)
            
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )
        
    def forward(self, x):
        out = F.gelu(self.bn1(self.linear1(x)))
        out = self.bn2(self.linear2(out))
        
        # 应用SE注意力
        if self.use_se:
            out = self.se(out)
            
        out += self.shortcut(x)
        out = F.gelu(out)
        return out

# 定义混合架构的生存预测模型 - 增强版
class HybridSurvivalPredictor(nn.Module):
    def __init__(self, input_size, hidden_size1=192, hidden_size2=96, output_size=3):
        super(HybridSurvivalPredictor, self).__init__()
        
        # 专为大规模特征设计的架构
        self.input_bn = nn.BatchNorm1d(input_size)
        
        # 增加隐藏层单元以增强网络容量
        self.res1 = ResidualBlock(input_size, hidden_size1, use_se=True)
        self.dropout1 = nn.Dropout(0.5)
        
        self.res2 = ResidualBlock(hidden_size1, hidden_size2, use_se=True)
        self.dropout2 = nn.Dropout(0.5)
        
        # 添加一个额外的残差层以增强表示能力
        self.res3 = ResidualBlock(hidden_size2, hidden_size2 // 2, use_se=True)
        self.dropout3 = nn.Dropout(0.5)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(hidden_size2 // 2, output_size),
            nn.BatchNorm1d(output_size)
        )
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.input_bn(x)
        x = self.res1(x)
        x = self.dropout1(x)
        
        x = self.res2(x)
        x = self.dropout2(x)
        
        x = self.res3(x)
        x = self.dropout3(x)
        
        x = self.output(x)
        
        return x

# 损失函数定义
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight if weight is not None else torch.FloatTensor(TRAINING_PARAMS['class_weights'])
        
    def forward(self, input, target):
        return F.cross_entropy(input, target, weight=self.weight)
