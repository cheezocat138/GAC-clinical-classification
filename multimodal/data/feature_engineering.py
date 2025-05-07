"""
特征工程模块

提供高级特征选择和特征工程方法，用于提高模型性能。
包括特征选择、特征交互、降维等技术。
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, chi2, f_classif,
    RFE, SelectFromModel, VarianceThreshold
)
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso, LassoCV, ElasticNet, ElasticNetCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import FeatureAgglomeration

from multimodal.utils.logger import get_logger

# 获取日志记录器
logger = get_logger("FeatureEngineering")


class AdvancedFeatureSelector:
    """高级特征选择器类"""
    
    def __init__(self, method='rfe', n_features=1000, random_state=42):
        """
        初始化特征选择器
        
        参数:
            method (str): 特征选择方法
            n_features (int): 要选择的特征数量
            random_state (int): 随机种子
        """
        self.method = method
        self.n_features = n_features
        self.random_state = random_state
        self.selector = None
        
    def fit_transform(self, X, y=None):
        """
        拟合并转换数据
        
        参数:
            X (numpy.ndarray): 特征矩阵
            y (numpy.ndarray): 目标变量
            
        返回:
            numpy.ndarray: 选择后的特征矩阵
        """
        if X.shape[1] <= self.n_features:
            logger.info(f"特征数量（{X.shape[1]}）已经小于等于目标数量（{self.n_features}），跳过特征选择")
            return X
            
        logger.info(f"应用 {self.method} 特征选择...")
        
        if self.method == 'variance':
            # 方差阈值特征选择
            threshold = 0.01  # 方差阈值
            self.selector = VarianceThreshold(threshold=threshold)
            X_selected = self.selector.fit_transform(X)
            
            # 如果剩余特征过多，使用SelectKBest进一步选择
            if X_selected.shape[1] > self.n_features:
                second_selector = SelectKBest(mutual_info_classif, k=self.n_features)
                X_selected = second_selector.fit_transform(X_selected, y)
                
        elif self.method == 'mutual_info':
            # 互信息特征选择
            self.selector = SelectKBest(mutual_info_classif, k=min(self.n_features, X.shape[1]))
            X_selected = self.selector.fit_transform(X, y)
            
        elif self.method == 'chi2':
            # 卡方检验 (要求数据非负)
            # 确保数据非负
            X_positive = X - X.min(axis=0) if X.min() < 0 else X
            self.selector = SelectKBest(chi2, k=min(self.n_features, X.shape[1]))
            X_selected = self.selector.fit_transform(X_positive, y)
            
        elif self.method == 'f_classif':
            # F检验
            self.selector = SelectKBest(f_classif, k=min(self.n_features, X.shape[1]))
            X_selected = self.selector.fit_transform(X, y)
            
        elif self.method == 'rfe':
            # 递归特征消除
            estimator = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            self.selector = RFE(estimator, n_features_to_select=min(self.n_features, X.shape[1]), step=0.1)
            X_selected = self.selector.fit_transform(X, y)
            
        elif self.method == 'lasso':
            # Lasso特征选择
            alpha = 0.01
            self.selector = SelectFromModel(
                Lasso(alpha=alpha, random_state=self.random_state),
                max_features=min(self.n_features, X.shape[1])
            )
            X_selected = self.selector.fit_transform(X, y)
            
        elif self.method == 'elasticnet':
            # ElasticNet特征选择
            self.selector = SelectFromModel(
                ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=self.random_state),
                max_features=min(self.n_features, X.shape[1])
            )
            X_selected = self.selector.fit_transform(X, y)
            
        elif self.method == 'forest':
            # 随机森林特征重要性
            self.selector = SelectFromModel(
                RandomForestClassifier(n_estimators=100, random_state=self.random_state),
                max_features=min(self.n_features, X.shape[1])
            )
            X_selected = self.selector.fit_transform(X, y)
            
        elif self.method == 'svm':
            # 线性SVM特征选择
            self.selector = SelectFromModel(
                LinearSVC(C=0.01, penalty="l1", dual=False, random_state=self.random_state),
                max_features=min(self.n_features, X.shape[1])
            )
            X_selected = self.selector.fit_transform(X, y)
            
        else:
            logger.warning(f"未知的特征选择方法: {self.method}，使用原始特征")
            X_selected = X
            
        logger.info(f"特征选择: {X.shape[1]} -> {X_selected.shape[1]}")
        return X_selected
    
    def transform(self, X):
        """
        转换新数据
        
        参数:
            X (numpy.ndarray): 特征矩阵
            
        返回:
            numpy.ndarray: 选择后的特征矩阵
        """
        if self.selector is None:
            logger.warning("转换前必须先调用fit_transform")
            return X
        
        return self.selector.transform(X)


class FeatureEngineer:
    """特征工程类，提供特征交互、降维等功能"""
    
    def __init__(self, interaction=False, polynomial=False, pca=False):
        """
        初始化特征工程类
        
        参数:
            interaction (bool): 是否使用特征交互
            polynomial (bool): 是否使用多项式特征
            pca (bool): 是否使用PCA降维
        """
        self.interaction = interaction
        self.polynomial = polynomial
        self.pca = pca
        self.poly_features = None
        self.pca_model = None
        
    def fit_transform(self, X, y=None, n_components=0.95):
        """
        拟合并转换数据
        
        参数:
            X (numpy.ndarray): 特征矩阵
            y (numpy.ndarray): 目标变量
            n_components (float or int): PCA组件数量或解释方差比例
            
        返回:
            numpy.ndarray: 转换后的特征矩阵
        """
        X_transformed = X.copy()
        
        # 应用多项式特征
        if self.polynomial:
            logger.info("应用多项式特征...")
            self.poly_features = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            X_poly = self.poly_features.fit_transform(X)
            logger.info(f"多项式特征转换: {X.shape[1]} -> {X_poly.shape[1]}")
            X_transformed = X_poly
        
        # 应用PCA降维
        if self.pca:
            logger.info("应用PCA降维...")
            self.pca_model = PCA(n_components=n_components, random_state=42)
            X_pca = self.pca_model.fit_transform(X_transformed)
            logger.info(f"PCA降维: {X_transformed.shape[1]} -> {X_pca.shape[1]}")
            X_transformed = X_pca
            
        return X_transformed
    
    def transform(self, X):
        """
        转换新数据
        
        参数:
            X (numpy.ndarray): 特征矩阵
            
        返回:
            numpy.ndarray: 转换后的特征矩阵
        """
        X_transformed = X.copy()
        
        if self.polynomial and self.poly_features is not None:
            X_transformed = self.poly_features.transform(X_transformed)
            
        if self.pca and self.pca_model is not None:
            X_transformed = self.pca_model.transform(X_transformed)
            
        return X_transformed


def create_feature_interactions(df, columns=None, interaction_type='multiply'):
    """
    创建特征交互
    
    参数:
        df (pandas.DataFrame): 输入数据框
        columns (list): 要进行交互的列名
        interaction_type (str): 交互类型 'multiply', 'add', 'subtract', 'divide'
        
    返回:
        pandas.DataFrame: 添加了交互特征的数据框
    """
    if columns is None:
        columns = df.columns.tolist()
    
    result_df = df.copy()
    
    # 生成所有列对
    n = len(columns)
    for i in range(n):
        for j in range(i+1, n):
            col1 = columns[i]
            col2 = columns[j]
            
            if interaction_type == 'multiply':
                result_df[f"{col1}*{col2}"] = df[col1] * df[col2]
            elif interaction_type == 'add':
                result_df[f"{col1}+{col2}"] = df[col1] + df[col2]
            elif interaction_type == 'subtract':
                result_df[f"{col1}-{col2}"] = df[col1] - df[col2]
            elif interaction_type == 'divide':
                # 避免除以0
                result_df[f"{col1}/{col2}"] = df[col1] / (df[col2] + 1e-8)
    
    return result_df


def auto_feature_engineering(X, y, method='auto', n_features=1000):
    """
    自动特征工程
    
    参数:
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 目标变量
        method (str): 特征工程方法
        n_features (int): 最终特征数量
        
    返回:
        numpy.ndarray: 工程化后的特征矩阵
    """
    logger.info(f"应用自动特征工程: {method}...")
    
    if method == 'auto':
        # 使用Lasso选择特征
        lasso_selector = SelectFromModel(
            LassoCV(cv=5, random_state=42), 
            max_features=n_features
        )
        X_selected = lasso_selector.fit_transform(X, y)
        
        # 添加一些多项式特征
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_poly = poly.fit_transform(X_selected)
        
        # 再次选择特征
        final_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=42),
            max_features=n_features
        )
        X_final = final_selector.fit_transform(X_poly, y)
        
        return X_final
    
    elif method == 'cluster':
        # 使用特征聚类
        agglo = FeatureAgglomeration(n_clusters=n_features)
        X_cluster = agglo.fit_transform(X, y)
        return X_cluster
    
    elif method == 'kernel_pca':
        # 使用核PCA
        kpca = KernelPCA(n_components=n_features, kernel='rbf', random_state=42)
        X_kpca = kpca.fit_transform(X)
        return X_kpca
        
    else:
        logger.warning(f"未知的特征工程方法: {method}，返回原始特征")
        return X 