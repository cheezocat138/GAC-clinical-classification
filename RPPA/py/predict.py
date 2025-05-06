import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from models import HybridSurvivalPredictor
from utils import ensure_dir, add_cluster_features
from torch.serialization import add_safe_globals
from sklearn.preprocessing._data import RobustScaler
from feature_processor import FeatureProcessor
from config import MODEL_PATHS, DATA_PATHS, OUTPUT_DIRS, PREDICTION_PATH, CLUSTER_PARAMS

# 添加安全全局变量
add_safe_globals([RobustScaler, FeatureProcessor])

def load_model(model_path=MODEL_PATHS['ensemble']):
    """加载预训练的生存预测模型"""
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载模型，设置weights_only=False以支持加载非张量对象
    try:
        checkpoint = torch.load(model_path, weights_only=False)
    except Exception as e:
        # 如果加载失败，尝试使用安全全局变量加载
        print(f"首次加载模型失败: {e}")
        print("尝试使用weights_only=False加载模型...")
        checkpoint = torch.load(model_path, weights_only=False)
    
    # 提取必要的模型信息
    if 'fold_models' in checkpoint:
        # 这是集成模型
        print("加载集成模型...")
        models = []
        input_size = checkpoint.get('input_shape')
        
        if not input_size or input_size == 0:
            # 从第一个模型状态字典推断输入大小
            model_state = checkpoint['fold_models'][0]
            for name, param in model_state.items():
                if 'input_bn.weight' in name:
                    input_size = param.size(0)
                    break
            
            if not input_size or input_size == 0:
                raise ValueError("无法从模型状态中确定输入大小")
        
        print(f"推断出模型输入大小: {input_size}")
        
        for i, model_state in enumerate(checkpoint['fold_models']):
            model = HybridSurvivalPredictor(input_size)
            model.load_state_dict(model_state)
            model.eval()
            models.append(model)
        
        # 提取特征处理相关信息
        scaler = checkpoint.get('scaler')
        selected_features = checkpoint.get('selected_features', [])
        selected_indices = checkpoint.get('selected_indices', [])
        feature_pipeline = checkpoint.get('feature_pipeline')
        
        # 修复：确保selected_indices不是空列表，如果是None或空列表则使用全部特征
        if selected_indices is None or len(selected_indices) == 0:
            print("没有特征选择信息，将使用所有特征")
            selected_indices = list(range(min(60, 227)))  # 使用前60个特征
        
        # 如果没有保存特征处理管道，则创建一个
        if feature_pipeline is None:
            print("模型中没有保存特征处理管道，创建新的处理器...")
            feature_processor = FeatureProcessor(selected_indices=selected_indices)
        else:
            feature_processor = feature_pipeline
            
        # 记录维度信息
        original_shape = checkpoint.get('original_shape')
        selected_shape = checkpoint.get('selected_shape')
        enhanced_shape = checkpoint.get('enhanced_shape')
        
        print(f"特征维度信息: 原始={original_shape}, 选择后={selected_shape}, 增强后={enhanced_shape}")
        
        return {
            'models': models,
            'scaler': scaler,
            'selected_features': selected_features,
            'selected_indices': selected_indices,
            'feature_processor': feature_processor,
            'is_ensemble': True,
            'input_size': input_size
        }
    else:
        # 单一模型
        print("加载单一模型...")
        model_state_dict = checkpoint.get('model_state_dict')
        if not model_state_dict:
            raise ValueError("无法从检查点中加载模型状态")
        
        # 从模型状态字典推断输入大小
        input_size = 0
        for name, param in model_state_dict.items():
            if 'input_bn.weight' in name:
                input_size = param.size(0)
                break
        
        if input_size == 0:
            raise ValueError("无法确定模型输入大小")
        
        print(f"推断出模型输入大小: {input_size}")
        
        # 创建正确尺寸的模型
        model = HybridSurvivalPredictor(input_size)
        model.load_state_dict(model_state_dict)
        model.eval()
        
        # 提取特征处理相关信息
        scaler = checkpoint.get('scaler')
        selected_features = checkpoint.get('selected_features', [])
        selected_indices = checkpoint.get('selected_indices', [])
        feature_pipeline = checkpoint.get('feature_pipeline')
        
        # 修复：确保selected_indices不是空列表，如果是None或空列表则使用全部特征
        if selected_indices is None or len(selected_indices) == 0:
            print("没有特征选择信息，将使用所有特征")
            selected_indices = list(range(min(60, 227)))  # 使用前60个特征
        
        # 如果没有保存特征处理管道，则创建一个
        if feature_pipeline is None:
            print("模型中没有保存特征处理管道，创建新的处理器...")
            feature_processor = FeatureProcessor(selected_indices=selected_indices)
        else:
            feature_processor = feature_pipeline
            
        # 记录维度信息
        original_shape = checkpoint.get('original_shape')
        selected_shape = checkpoint.get('selected_shape')
        enhanced_shape = checkpoint.get('enhanced_shape')
        
        print(f"特征维度信息: 原始={original_shape}, 选择后={selected_shape}, 增强后={enhanced_shape}")
        
        return {
            'model': model,
            'scaler': scaler,
            'selected_features': selected_features,
            'selected_indices': selected_indices,
            'feature_processor': feature_processor,
            'is_ensemble': False,
            'input_size': input_size
        }

def predict_survival_group(data, model_info, return_probabilities=False):
    """使用预训练模型预测生存组别"""
    print(f"原始数据特征数量: {data.shape[1]}")
    
    # 提取特征，如果是DataFrame就转为numpy数组
    if isinstance(data, pd.DataFrame):
        # 如果有选定特征列表，使用它们
        if model_info['selected_features'] and len(model_info['selected_features']) > 0:
            print(f"使用{len(model_info['selected_features'])}个选定特征")
            # 检查是否所有特征都在数据中
            available_features = set(data.columns)
            selected_features = [f for f in model_info['selected_features'] if f in available_features]
            
            if len(selected_features) < len(model_info['selected_features']):
                missing = len(model_info['selected_features']) - len(selected_features)
                print(f"警告: 数据中缺少{missing}个特征. 使用可用特征子集.")
            
            if not selected_features:
                print("警告：没有可用的特征进行预测，将使用所有特征")
                features = data.values
            else:
                features = data[selected_features].values
            
        elif model_info['selected_indices'] is not None and len(model_info['selected_indices']) > 0:
            # 如果有特征索引，先检查是否都在范围内
            print(f"使用{len(model_info['selected_indices'])}个特征索引")
            valid_indices = [i for i in model_info['selected_indices'] if i < data.shape[1]]
            
            if len(valid_indices) < len(model_info['selected_indices']):
                missing = len(model_info['selected_indices']) - len(valid_indices)
                print(f"警告: {missing}个特征索引超出范围. 使用可用索引子集.")
                
            if not valid_indices:
                print("警告：没有有效的特征索引进行预测，将使用所有特征")
                features = data.values
            else:
                # 使用有效的特征索引
                features = data.iloc[:, valid_indices].values
        else:
            # 没有特征选择信息
            print("没有特征选择信息，使用全部特征")
            features = data.values
    else:
        # 假设已经是numpy数组
        features = data
    
    print(f"提取的初始特征形状: {features.shape}")
    
    # 防止empty feature array
    if features.shape[1] == 0:
        print("警告: 特征数量为0，使用原始数据")
        if isinstance(data, pd.DataFrame):
            features = data.values
        else:
            features = data
        print(f"使用原始数据，特征形状: {features.shape}")
    
    # 应用特征处理器 - 使用一致的特征选择和转换流程
    try:
        print("应用特征处理器...")
        features = model_info['feature_processor'].transform(features)
        print(f"处理后特征形状: {features.shape}")
    except Exception as e:
        print(f"特征处理器应用失败: {e}")
        # 如果失败，直接应用聚类特征
        print("尝试直接应用聚类特征...")
        features = add_cluster_features(features)
        print(f"添加聚类特征后形状: {features.shape}")
    
    # 使用模型的缩放器进行特征标准化
    if model_info['scaler'] is not None:
        try:
            print("应用特征标准化...")
            features = model_info['scaler'].transform(features)
        except ValueError as e:
            print(f"特征标准化失败: {e}")
            # 如果失败，尝试重新拟合一个scaler
            from sklearn.preprocessing import RobustScaler
            print("创建新的scaler...")
            new_scaler = RobustScaler()
            features = new_scaler.fit_transform(features)
    
    # 确保特征数量与模型期望一致
    if features.shape[1] != model_info['input_size']:
        raise ValueError(f"特征数量不匹配: 当前={features.shape[1]}, 模型期望={model_info['input_size']}")
    
    # 转换为PyTorch张量
    features_tensor = torch.FloatTensor(features)
    
    # 使用模型进行预测
    if model_info['is_ensemble']:
        # 集成预测
        all_probabilities = []
        
        for model in model_info['models']:
            with torch.no_grad():
                outputs = model(features_tensor)
                probs = F.softmax(outputs, dim=1)
                all_probabilities.append(probs.numpy())
        
        # 平均所有模型的预测概率
        all_probabilities = np.array(all_probabilities)
        avg_probabilities = np.mean(all_probabilities, axis=0)
        predictions = np.argmax(avg_probabilities, axis=1) + 1  # 转换回1-3编码
        
        if return_probabilities:
            return predictions, avg_probabilities
        else:
            return predictions
    else:
        # 单模型预测
        model = model_info['model']
        with torch.no_grad():
            outputs = model(features_tensor)
            probs = F.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1).numpy() + 1  # 转换回1-3编码
        
        if return_probabilities:
            return predictions, probs.numpy()
        else:
            return predictions

def predict_from_rppa(rppa_file=DATA_PATHS['rppa_transposed'], model_path=MODEL_PATHS['ensemble'], output_file=PREDICTION_PATH):
    """从RPPA数据文件预测生存组别"""
    # 加载RPPA数据
    rppa_data = pd.read_csv(rppa_file)
    sample_ids = rppa_data['sample'].values
    rppa_features = rppa_data.drop('sample', axis=1)
    
    # 处理缺失值
    rppa_features = rppa_features.fillna(rppa_features.median())
    print(f"RPPA数据形状: {rppa_features.shape}")
    
    # 加载模型
    model_info = load_model(model_path)
    
    # 进行预测
    predictions, probabilities = predict_survival_group(rppa_features, model_info, return_probabilities=True)
    
    # 创建预测结果DataFrame
    result_df = pd.DataFrame({
        'sample': sample_ids,
        'predicted_group': predictions
    })
    
    # 添加每个类别的预测概率
    for i in range(probabilities.shape[1]):
        result_df[f'prob_group_{i+1}'] = probabilities[:, i]
    
    # 保存结果
    if output_file:
        ensure_dir(os.path.dirname(output_file))
        result_df.to_csv(output_file, index=False)
        print(f"预测结果已保存到: {output_file}")
    
    return result_df
