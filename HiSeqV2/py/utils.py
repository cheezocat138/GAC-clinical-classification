import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.metrics import confusion_matrix

from config import (TRAINING_PARAMS, FEATURE_SELECTION_PARAMS, 
                   CLUSTER_PARAMS, PLOT_PARAMS)

# 确保目录存在
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# PyTorch数据集定义
class RPPADataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 梯度裁剪工具函数
def clip_gradients(model, clip_value=0.5):  
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

# 循环退火学习率策略
def cyclical_lr(initial_lr, max_lr, min_lr, stepsize):
    """循环退火学习率生成器，修正版"""
    def lr_lambda(it):
        cycle = np.floor(1 + it / (2 * stepsize))
        x = np.abs(it / stepsize - 2 * cycle + 1)
        lr_factor = min_lr + (max_lr - min_lr) * max(0, (1 - x)) / initial_lr
        return lr_factor
    
    return lr_lambda

# 增强型特征选择策略 - 为大规模数据优化
def enhanced_feature_selection(X, y):
    """针对HiSeqV2大规模数据优化的特征选择策略"""
    print("执行增强型特征选择，优化用于大规模数据...")
    
    # 对于非常大的特征集，我们可能需要进行分批处理
    max_features = FEATURE_SELECTION_PARAMS['mi_k']
    
    # 1. 使用互信息选择特征 - 先进行快速预筛选
    print("第1步：使用互信息进行特征预筛选...")
    # 如果特征数量过大，则增加预筛选步骤
    if X.shape[1] > 5000:
        # 先用更快的方法预筛选特征
        pre_selector = SelectKBest(f_classif, k=min(3000, X.shape[1]))
        X_pre = pre_selector.fit_transform(X, y)
        pre_selected = pre_selector.get_support(indices=True)
        print(f"预筛选后特征数量: {len(pre_selected)}")
        
        # 然后在预筛选结果上应用互信息选择
        mi_selector = SelectKBest(mutual_info_classif, k=FEATURE_SELECTION_PARAMS['mi_k'])
        _ = mi_selector.fit_transform(X_pre, y)
        
        # 获取原始特征索引
        mi_selected_subset = mi_selector.get_support(indices=True)
        mi_selected = pre_selected[mi_selected_subset]
    else:
        # 直接应用互信息
        mi_selector = SelectKBest(mutual_info_classif, k=FEATURE_SELECTION_PARAMS['mi_k'])
        _ = mi_selector.fit_transform(X, y)
        mi_selected = mi_selector.get_support(indices=True)
    
    print(f"互信息选择后特征数量: {len(mi_selected)}")
    
    # 2. 使用ANOVA选择特征
    print("第2步：使用ANOVA进行特征选择...")
    f_selector = SelectKBest(f_classif, k=FEATURE_SELECTION_PARAMS['f_k'])
    _ = f_selector.fit_transform(X, y)
    f_selected = f_selector.get_support(indices=True)
    print(f"ANOVA选择后特征数量: {len(f_selected)}")
    
    # 3. 使用随机森林递归特征消除(RFE) - 仅在特征数量适中时使用
    print("第3步：使用随机森林特征重要性评估...")
    # 对于大规模特征，直接使用随机森林的特征重要性更高效
    rf_basic = RandomForestClassifier(
        n_estimators=100, 
        random_state=TRAINING_PARAMS['random_seed'], 
        class_weight='balanced',
        n_jobs=-1,  # 并行处理
        max_features='sqrt'  # 使用sqrt策略减少内存使用
    )
    
    # 如果特征数量过大，先使用前两种方法的结果
    if X.shape[1] > 5000:
        # 合并互信息和ANOVA的结果
        combined_indices = np.union1d(mi_selected, f_selected)
        print(f"互信息和ANOVA合并后特征数量: {len(combined_indices)}")
        
        # 使用这些特征训练随机森林
        rf_basic.fit(X[:, combined_indices], y)
        
        # 获取特征重要性
        importances = rf_basic.feature_importances_
        
        # 选择重要性最高的特征
        sorted_indices = np.argsort(importances)[::-1][:FEATURE_SELECTION_PARAMS['rfe_k']]
        
        # 转换回原始特征索引
        rfe_selected = combined_indices[sorted_indices]
    else:
        # 对于较小的特征集，可以使用RFE
        rfe = RFE(
            estimator=rf_basic, 
            n_features_to_select=FEATURE_SELECTION_PARAMS['rfe_k'], 
            step=max(10, X.shape[1] // 100)  # 较大步长以加速过程
        )
        rfe.fit(X, y)
        rfe_selected = np.where(rfe.support_)[0]
    
    print(f"随机森林/RFE选择后特征数量: {len(rfe_selected)}")
    
    # 4. 结合三种方法选择的特征
    print("第4步：合并特征选择结果...")
    # 计算特征出现在不同选择器中的次数
    feature_counts = np.zeros(X.shape[1])
    for idx in mi_selected:
        feature_counts[idx] += 1
    for idx in f_selected:
        feature_counts[idx] += 1
    for idx in rfe_selected:
        feature_counts[idx] += 1
        
    # 优先选择出现在多个选择器中的特征
    combined_features = np.where(feature_counts >= 2)[0]  # 至少在2个选择器中出现
    print(f"出现在至少2个选择器中的特征数量: {len(combined_features)}")
    
    # 如果特征太少，再添加一些重要特征
    if len(combined_features) < FEATURE_SELECTION_PARAMS['min_features']:
        # 按重要性排序添加更多特征
        remaining_features = np.where(feature_counts == 1)[0]
        
        # 使用随机森林评估剩余特征的重要性
        if len(remaining_features) > 0:
            print("特征数量不足，添加额外特征...")
            if len(remaining_features) > 3000:
                # 如果剩余特征太多，随机抽样
                np.random.seed(TRAINING_PARAMS['random_seed'])
                sample_size = min(3000, len(remaining_features))
                sampled_features = np.random.choice(remaining_features, size=sample_size, replace=False)
                remaining_features = sampled_features
            
            temp_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            temp_rf.fit(X[:, remaining_features], y)
            importances = temp_rf.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            
            # 添加足够的特征以达到最小特征数
            features_to_add = min(FEATURE_SELECTION_PARAMS['min_features'] - len(combined_features), len(sorted_indices))
            additional_features = remaining_features[sorted_indices[:features_to_add]]
            combined_features = np.append(combined_features, additional_features)
            print(f"添加了{features_to_add}个额外特征")
    
    print(f"最终选择的特征数量: {len(combined_features)}")
    
    return X[:, combined_features], combined_features

# 添加KMeans到安全全局变量列表
from torch.serialization import add_safe_globals
add_safe_globals([KMeans])

# 修改add_cluster_features，保存聚类模型方便重用
def add_cluster_features(X, n_clusters=CLUSTER_PARAMS['n_clusters'], kmeans_model=None):
    """添加基于聚类的特征"""
    print("添加基于聚类的辅助特征...")
    
    if kmeans_model is None:
        # 对于大规模数据，使用小批量聚类
        if X.shape[0] > 1000 or X.shape[1] > 1000:
            from sklearn.cluster import MiniBatchKMeans
            print("使用MiniBatchKMeans进行大规模数据聚类...")
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters, 
                random_state=CLUSTER_PARAMS['random_state'], 
                batch_size=100,
                n_init='auto'
            )
        else:
            kmeans = KMeans(
                n_clusters=n_clusters, 
                random_state=CLUSTER_PARAMS['random_state'], 
                n_init='auto'
            )
        kmeans.fit(X)
    else:
        kmeans = kmeans_model
        
    # 计算每个样本到各聚类中心的距离
    distances = kmeans.transform(X)
    
    # 将聚类标签作为新特征添加到原始特征
    X_with_clusters = np.column_stack((X, distances))
    
    print(f"添加聚类特征前维度: {X.shape}, 添加后: {X_with_clusters.shape}")
    
    return X_with_clusters

# 绘制混淆矩阵
def plot_confusion_matrix(y_true, y_pred, classes=PLOT_PARAMS['group_names'], output_path=None, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=PLOT_PARAMS['figsize_small'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    if output_path:
        plt.savefig(output_path, dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
    
    plt.close()
