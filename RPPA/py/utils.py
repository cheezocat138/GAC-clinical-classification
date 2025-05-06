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

# 增强型特征选择策略
def enhanced_feature_selection(X, y):
    print("执行增强型特征选择...")
    
    # 1. 使用互信息选择特征
    mi_selector = SelectKBest(mutual_info_classif, k=FEATURE_SELECTION_PARAMS['mi_k'])
    X_mi = mi_selector.fit_transform(X, y)
    mi_selected = mi_selector.get_support(indices=True)
    
    # 2. 使用ANOVA选择特征
    f_selector = SelectKBest(f_classif, k=FEATURE_SELECTION_PARAMS['f_k'])
    X_f = f_selector.fit_transform(X, y)
    f_selected = f_selector.get_support(indices=True)
    
    # 3. 使用随机森林递归特征消除(RFE)
    rf_basic = RandomForestClassifier(n_estimators=200, random_state=TRAINING_PARAMS['random_seed'], class_weight='balanced')
    rfe = RFE(estimator=rf_basic, n_features_to_select=FEATURE_SELECTION_PARAMS['rfe_k'], step=10)
    rfe.fit(X, y)
    rfe_selected = np.where(rfe.support_)[0]
    
    # 4. 结合三种方法选择的特征（更倾向于多方法共同选择的特征）
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
    
    # 如果特征太少，再添加一些重要特征
    if len(combined_features) < FEATURE_SELECTION_PARAMS['min_features']:
        # 按重要性排序添加更多特征
        remaining_features = np.where(feature_counts == 1)[0]
        
        # 使用随机森林评估剩余特征的重要性
        if len(remaining_features) > 0:
            temp_rf = RandomForestClassifier(n_estimators=100, random_state=42)
            temp_rf.fit(X[:, remaining_features], y)
            importances = temp_rf.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            
            # 添加足够的特征以达到最小特征数
            features_to_add = min(60 - len(combined_features), len(sorted_indices))
            additional_features = remaining_features[sorted_indices[:features_to_add]]
            combined_features = np.append(combined_features, additional_features)
    
    print(f"特征选择结果: MI={len(mi_selected)}, ANOVA={len(f_selected)}, RFE={len(rfe_selected)}, 组合后={len(combined_features)}")
    
    return X[:, combined_features], combined_features

# 添加KMeans到安全全局变量列表
from torch.serialization import add_safe_globals
add_safe_globals([KMeans])

# 修改add_cluster_features，保存聚类模型方便重用
def add_cluster_features(X, n_clusters=CLUSTER_PARAMS['n_clusters'], kmeans_model=None):
    """添加基于聚类的特征"""
    print("添加基于聚类的辅助特征...")
    
    if kmeans_model is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=CLUSTER_PARAMS['random_state'], n_init=10)
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
