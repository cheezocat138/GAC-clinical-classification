import sys
import traceback
sys.path.append('.')

import torch
import numpy as np
from multimodal.models.fusion_model import (
    EarlyFusionModel,
    LateFusionModel,
    AttentionFusionModel,
    CrossModalTransformerFusion,
    HybridMultiModalNet
)

def test_fusion_models():
    """测试不同类型的融合模型"""
    try:
        print("创建测试数据...")
        batch_size = 16
        hiseq_dim = 1000  # 与实际数据集维度类似
        rppa_dim = 225    # 与实际数据集维度类似
        
        hiseq_data = torch.randn(batch_size, hiseq_dim)
        rppa_data = torch.randn(batch_size, rppa_dim)
        
        # 测试设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 模型参数
        hidden_size = 256
        dropout_rate = 0.3
        num_classes = 3
        
        # 测试各种融合模型
        print("\n测试早期融合模型...")
        early_model = EarlyFusionModel(hiseq_dim, rppa_dim, hidden_size, dropout_rate, num_classes).to(device)
        hiseq_data_device = hiseq_data.to(device)
        rppa_data_device = rppa_data.to(device)
        early_output = early_model(hiseq_data_device, rppa_data_device)
        print(f"模型结构:\n{early_model}")
        print(f"输入形状: HiSeq={hiseq_data.shape}, RPPA={rppa_data.shape}")
        print(f"输出形状: {early_output.shape}")
        
        print("\n测试晚期融合模型...")
        late_model = LateFusionModel(hiseq_dim, rppa_dim, hidden_size, dropout_rate, num_classes).to(device)
        late_output, hiseq_out, rppa_out = late_model(hiseq_data_device, rppa_data_device)
        print(f"融合输出形状: {late_output.shape}")
        print(f"HiSeq输出形状: {hiseq_out.shape}")
        print(f"RPPA输出形状: {rppa_out.shape}")
        print(f"融合权重: {late_model.fusion_weights.data}")
        
        print("\n测试注意力融合模型...")
        attention_model = AttentionFusionModel(hiseq_dim, rppa_dim, hidden_size, dropout_rate, num_classes).to(device)
        attention_output = attention_model(hiseq_data_device, rppa_data_device)
        print(f"输出形状: {attention_output.shape}")
        
        print("\n测试跨模态Transformer融合模型...")
        transformer_model = CrossModalTransformerFusion(hiseq_dim, rppa_dim, hidden_size, num_heads=4, num_layers=2, dropout_rate=dropout_rate, num_classes=num_classes).to(device)
        transformer_output = transformer_model(hiseq_data_device, rppa_data_device)
        print(f"输出形状: {transformer_output.shape}")
        
        print("\n测试混合多模态网络...")
        hybrid_model = HybridMultiModalNet(hiseq_dim, rppa_dim, hidden_size, dropout_rate, num_classes).to(device)
        hybrid_output = hybrid_model(hiseq_data_device, rppa_data_device)
        print(f"输出形状: {hybrid_output.shape}")
        print(f"融合权重: {hybrid_model.fusion_weights.data}")
        
        # 计算模型参数量
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print("\n模型参数数量比较:")
        print(f"早期融合模型: {count_parameters(early_model):,} 参数")
        print(f"晚期融合模型: {count_parameters(late_model):,} 参数")
        print(f"注意力融合模型: {count_parameters(attention_model):,} 参数")
        print(f"Transformer融合模型: {count_parameters(transformer_model):,} 参数")
        print(f"混合多模态网络: {count_parameters(hybrid_model):,} 参数")
        
        # 测试向后传播
        print("\n测试反向传播...")
        criterion = torch.nn.CrossEntropyLoss()
        targets = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        # 早期融合
        optimizer = torch.optim.Adam(early_model.parameters(), lr=0.001)
        optimizer.zero_grad()
        outputs = early_model(hiseq_data_device, rppa_data_device)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"早期融合模型反向传播成功，损失: {loss.item():.4f}")
        
        print("\n所有模型测试完成!")
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("开始测试融合模型...")
    success = test_fusion_models()
    print(f"测试{'成功' if success else '失败'}!") 