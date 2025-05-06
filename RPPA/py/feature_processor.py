import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from utils import add_cluster_features
from config import CLUSTER_PARAMS

class FeatureProcessor:
    """特征处理类，确保训练和预测时使用一致的特征处理流程"""
    
    def __init__(self, selected_indices=None, n_clusters=CLUSTER_PARAMS['n_clusters']):
        """初始化特征处理器
        
        参数:
        selected_indices -- 特征选择索引
        n_clusters -- 聚类数量
        """
        self.selected_indices = selected_indices
        self.n_clusters = n_clusters
        self.pipeline = None
        self._fitted = False
        
    def fit(self, X):
        """拟合特征处理流程
        
        参数:
        X -- 输入特征矩阵
        """
        # 创建特征处理管道
        steps = []
        
        # 如果有特征选择，添加特征选择步骤
        if self.selected_indices is not None and len(self.selected_indices) > 0:
            steps.append(('feature_selection', 
                         FunctionTransformer(lambda x: x[:, self.selected_indices])))
        
        # 添加聚类特征
        steps.append(('add_clusters', 
                     FunctionTransformer(lambda x: add_cluster_features(x, self.n_clusters))))
        
        self.pipeline = Pipeline(steps)
        self.pipeline.fit(X)
        self._fitted = True
        return self
        
    def transform(self, X):
        """应用特征处理
        
        参数:
        X -- 输入特征矩阵
        
        返回:
        处理后的特征矩阵
        """
        if not self._fitted:
            # 如果管道未拟合，先使用当前数据拟合它
            print("特征处理器未拟合，使用当前数据拟合")
            self.fit(X)
            
        # 安全处理：直接应用特征选择和聚类
        transformed_X = X
        
        # 如果有选择的特征列，先应用特征选择
        if self.selected_indices is not None and len(self.selected_indices) > 0:
            # 确保索引在有效范围内
            valid_indices = [idx for idx in self.selected_indices if idx < X.shape[1]]
            if len(valid_indices) > 0:
                transformed_X = X[:, valid_indices]
                print(f"应用特征选择，从 {X.shape[1]} 维减少到 {transformed_X.shape[1]} 维")
            else:
                print("警告：没有有效的特征索引，保持原始特征")
        
        # 添加聚类特征
        if transformed_X.shape[1] > 0:
            transformed_X = add_cluster_features(transformed_X, self.n_clusters)
            print(f"添加聚类特征后，特征维度扩展到 {transformed_X.shape[1]}")
        else:
            print("错误：没有特征可用于聚类，无法继续处理")
            # 紧急恢复：使用原始特征
            transformed_X = add_cluster_features(X, self.n_clusters)
            print(f"使用全部原始特征 ({X.shape[1]} 维) 添加聚类特征，最终维度为 {transformed_X.shape[1]}")
        
        return transformed_X
    
    def fit_transform(self, X):
        """拟合并应用特征处理
        
        参数:
        X -- 输入特征矩阵
        
        返回:
        处理后的特征矩阵
        """
        return self.fit(X).transform(X)
    
    def get_output_shape(self, input_shape):
        """获取输出特征的形状
        
        参数:
        input_shape -- 输入特征的形状
        
        返回:
        处理后特征的形状
        """
        selected_shape = len(self.selected_indices) if self.selected_indices else input_shape
        # 聚类特征添加的维度是聚类数量
        return selected_shape + self.n_clusters
