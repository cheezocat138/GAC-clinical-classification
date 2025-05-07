"""
多模态数据集模块

该模块提供用于处理和整合HiSeq和RPPA数据的功能。
主要实现:
1. 多模态数据加载
2. 共同样本匹配
3. 特征选择与预处理
4. 数据集创建
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif
from imblearn.over_sampling import SMOTE

from multimodal.utils.config import CONFIG
from multimodal.utils.logger import get_logger
from multimodal.data.hiseq_data import HiSeqDataProcessor
from multimodal.data.rppa_data import RPPADataProcessor

# 获取日志记录器
logger = get_logger("MultiModalData")

class MultiModalDataset(Dataset):
    """多模态数据集类"""
    
    def __init__(self, hiseq_features, rppa_features, labels):
        """
        初始化多模态数据集
        
        参数:
            hiseq_features (numpy.ndarray): HiSeq特征
            rppa_features (numpy.ndarray): RPPA特征
            labels (numpy.ndarray): 标签
        """
        self.hiseq_features = torch.FloatTensor(hiseq_features)
        self.rppa_features = torch.FloatTensor(rppa_features)
        # 确保标签值从0开始
        self.labels = torch.LongTensor(labels.astype(int) - 1)
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """获取指定索引的样本"""
        return self.hiseq_features[idx], self.rppa_features[idx], self.labels[idx]


class MultiModalDataProcessor:
    """多模态数据处理器"""
    
    def __init__(self, config=None):
        """
        初始化数据处理器
        
        参数:
            config (dict): 配置字典
        """
        self.config = config if config is not None else CONFIG
        self.hiseq_processor = HiSeqDataProcessor(config=self.config)
        self.rppa_processor = RPPADataProcessor(config=self.config)
        
        # 初始化预处理对象
        self.hiseq_scaler = None
        self.rppa_scaler = None
        self.feature_selector = None
        
        # 创建数据目录
        os.makedirs(self.config['DATA_PATHS']['processed_data_dir'], exist_ok=True)
    
    def _align_samples(self, hiseq_df, rppa_df):
        """
        对齐HiSeq和RPPA样本
        
        参数:
            hiseq_df (pandas.DataFrame): HiSeq数据框
            rppa_df (pandas.DataFrame): RPPA数据框
            
        返回:
            tuple: (hiseq_df, rppa_df)
        """
        logger.info("对齐HiSeq和RPPA样本...")
        
        # 获取样本ID
        hiseq_ids = list(hiseq_df.index)  # 转换为列表
        
        # RPPA数据可能将样本ID存储为一列
        if 'sample' in rppa_df.columns:
            rppa_sample_col = 'sample'
            # 将样本ID设置为索引
            rppa_df = rppa_df.set_index(rppa_sample_col)
        
        rppa_ids = list(rppa_df.index)  # 转换为列表
        
        logger.info(f"HiSeq样本数量: {len(hiseq_ids)}")
        logger.info(f"RPPA样本数量: {len(rppa_ids)}")
        
        # 尝试直接匹配
        common_samples_set = set(hiseq_ids).intersection(set(rppa_ids))
        common_samples = list(common_samples_set)  # 确保转换为列表
        logger.info(f"直接匹配到{len(common_samples)}个共同样本")
        
        # 如果直接匹配不成功，尝试简化样本ID后匹配
        if len(common_samples) == 0:
            logger.info("尝试简化样本ID后匹配...")
            
            # 简化样本ID (例如: TCGA-XX-YYYY-01 -> TCGA-XX-YYYY)
            hiseq_ids_simple = {}
            rppa_ids_simple = {}
            
            for id in hiseq_ids:
                # 尝试提取TCGA-XX-YYYY部分
                parts = str(id).split('-')  # 确保ID是字符串
                if len(parts) >= 3:
                    simple_id = '-'.join(parts[:3])
                    hiseq_ids_simple[simple_id] = id
            
            for id in rppa_ids:
                # 尝试提取TCGA-XX-YYYY部分
                parts = str(id).split('-')  # 确保ID是字符串
                if len(parts) >= 3:
                    simple_id = '-'.join(parts[:3])
                    rppa_ids_simple[simple_id] = id
            
            # 寻找简化后的共同ID
            common_simple_ids_set = set(hiseq_ids_simple.keys()).intersection(set(rppa_ids_simple.keys()))
            common_simple_ids = list(common_simple_ids_set)  # 确保转换为列表
            logger.info(f"简化后匹配到{len(common_simple_ids)}个共同样本")
            
            if len(common_simple_ids) > 0:
                # 映射回原始ID
                hiseq_common = [hiseq_ids_simple[simple_id] for simple_id in common_simple_ids]
                rppa_common = [rppa_ids_simple[simple_id] for simple_id in common_simple_ids]
                
                # 创建ID映射关系
                id_map = dict(zip(rppa_common, hiseq_common))
                
                # 确保索引为列表
                hiseq_df_aligned = hiseq_df.loc[hiseq_common]
                rppa_df_aligned = rppa_df.loc[rppa_common].copy()
                
                # 修改RPPA的索引以匹配HiSeq
                new_indices = [id_map[id] for id in rppa_df_aligned.index]
                rppa_df_aligned.index = new_indices
                
                logger.info(f"通过简化ID匹配成功，保留了{len(common_simple_ids)}个样本")
                return hiseq_df_aligned, rppa_df_aligned
            
            # 尝试更宽松的匹配策略
            if len(common_simple_ids) == 0:
                logger.info("尝试更宽松的匹配策略...")
                # 尝试提取TCGA-XX部分
                hiseq_ids_simple = {}
                rppa_ids_simple = {}
                
                for id in hiseq_ids:
                    parts = str(id).split('-')
                    if len(parts) >= 2:
                        simple_id = '-'.join(parts[:2])
                        hiseq_ids_simple[simple_id] = id
                
                for id in rppa_ids:
                    parts = str(id).split('-')
                    if len(parts) >= 2:
                        simple_id = '-'.join(parts[:2])
                        rppa_ids_simple[simple_id] = id
                
                common_simple_ids_set = set(hiseq_ids_simple.keys()).intersection(set(rppa_ids_simple.keys()))
                common_simple_ids = list(common_simple_ids_set)
                logger.info(f"使用更宽松匹配策略匹配到{len(common_simple_ids)}个共同样本")
                
                if len(common_simple_ids) > 0:
                    # 映射回原始ID
                    hiseq_common = [hiseq_ids_simple[simple_id] for simple_id in common_simple_ids]
                    rppa_common = [rppa_ids_simple[simple_id] for simple_id in common_simple_ids]
                    
                    # 创建ID映射关系
                    id_map = dict(zip(rppa_common, hiseq_common))
                    
                    # 确保索引为列表
                    hiseq_df_aligned = hiseq_df.loc[hiseq_common]
                    rppa_df_aligned = rppa_df.loc[rppa_common].copy()
                    
                    # 修改RPPA的索引以匹配HiSeq
                    new_indices = [id_map[id] for id in rppa_df_aligned.index]
                    rppa_df_aligned.index = new_indices
                    
                    logger.info(f"通过更宽松匹配策略成功，保留了{len(common_simple_ids)}个样本")
                    return hiseq_df_aligned, rppa_df_aligned
        
        if len(common_samples) == 0:
            logger.error("没有找到共同样本，请检查样本ID格式")
            return None, None
        
        # 过滤数据框
        hiseq_df = hiseq_df.loc[common_samples]
        rppa_df = rppa_df.loc[common_samples]
        
        # 确保样本顺序一致
        common_samples = sorted(common_samples)
        hiseq_df = hiseq_df.loc[common_samples]
        rppa_df = rppa_df.loc[common_samples]
        
        logger.info(f"样本对齐成功，保留了{len(common_samples)}个共同样本")
        return hiseq_df, rppa_df
    
    def _apply_feature_selection(self, X, k=1000, method='variance'):
        """
        应用特征选择
        
        参数:
            X (numpy.ndarray): 特征矩阵
            k (int): 要选择的特征数量
            method (str): 特征选择方法
            
        返回:
            numpy.ndarray: 选择后的特征矩阵
        """
        if method == 'variance':
            # 方差阈值选择
            self.feature_selector = VarianceThreshold(threshold=0.01)
            X_selected = self.feature_selector.fit_transform(X)
            
            # 如果特征太多，继续进行SelectKBest
            if X_selected.shape[1] > k:
                selector = SelectKBest(k=k)
                X_selected = selector.fit_transform(X_selected, self.y_train)
                
        elif method == 'mutual_info':
            # 互信息选择
            self.feature_selector = SelectKBest(mutual_info_classif, k=min(k, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, self.y_train)
            
        else:
            logger.warning(f"未知的特征选择方法: {method}，使用原始特征")
            X_selected = X
            
        logger.info(f"特征选择: {X.shape[1]} -> {X_selected.shape[1]}")
        return X_selected
    
    def load_data(self):
        """
        加载HiSeq和RPPA数据，并对齐样本
        
        返回:
            tuple: (hiseq_features, rppa_features, labels)
        """
        logger.info("加载HiSeq和RPPA数据...")
        
        # 加载预处理好的数据
        hiseq_df = self.hiseq_processor.load_data()
        rppa_df = self.rppa_processor.load_data()
        
        if 'survival_group_code' not in hiseq_df.columns:
            logger.error("HiSeq数据中缺少标签列，请确保数据中包含'survival_group_code'列")
            return None, None, None
            
        if 'survival_group_code' not in rppa_df.columns:
            logger.error("RPPA数据中缺少标签列，请确保数据中包含'survival_group_code'列")
            return None, None, None
            
        # 确保RPPA数据有正确的样本ID列
        rppa_sample_col = None
        if 'sample' in rppa_df.columns:
            rppa_sample_col = 'sample'
        
        # 分离特征和标签
        hiseq_features = hiseq_df.drop(columns=['survival_group_code'])
        
        if rppa_sample_col:
            # 如果有样本列，确保它不被删除
            rppa_features = rppa_df.drop(columns=[col for col in ['survival_group_code'] if col != rppa_sample_col])
        else:
            rppa_features = rppa_df.drop(columns=['survival_group_code'])
        
        # 对齐样本
        hiseq_features, rppa_features = self._align_samples(hiseq_features, rppa_features)
        
        if hiseq_features is None or rppa_features is None:
            return None, None, None
        
        # 获取标签
        # 由于对齐过程可能改变了索引，所以从HiSeq特征获取样本ID，再从rppa_df中找标签
        hiseq_indices = hiseq_features.index
        
        # 确保标签与样本对齐
        labels = hiseq_df.loc[hiseq_indices, 'survival_group_code'].values
        
        logger.info(f"加载完成。HiSeq特征: {hiseq_features.shape}, RPPA特征: {rppa_features.shape}, 标签: {labels.shape}")
        
        return hiseq_features.values, rppa_features.values, labels
    
    def preprocess_data(self, hiseq_features, rppa_features, labels):
        """
        预处理数据，包括缩放、特征选择等
        
        参数:
            hiseq_features (numpy.ndarray): HiSeq特征
            rppa_features (numpy.ndarray): RPPA特征
            labels (numpy.ndarray): 标签
            
        返回:
            tuple: (处理后的hiseq特征, 处理后的rppa特征, 处理后的标签)
        """
        logger.info("预处理数据...")
        
        # 保存训练数据（用于特征选择）
        self.X_hiseq = hiseq_features
        self.X_rppa = rppa_features
        self.y_train = labels
        
        # 应用HiSeq特征选择（仅当启用时）
        if self.config['DATA_PROCESSING']['feature_selection']['enabled'] and hiseq_features.shape[1] > self.config['DATA_PROCESSING']['feature_selection']['max_features']:
            hiseq_features = self._apply_feature_selection(
                hiseq_features,
                k=self.config['DATA_PROCESSING']['feature_selection']['max_features'],
                method=self.config['DATA_PROCESSING']['feature_selection']['method']
            )
        
        # 应用缩放
        hiseq_scaler_type = self.config['DATA_PROCESSING']['transformation']['hiseq']['scaler']
        rppa_scaler_type = self.config['DATA_PROCESSING']['transformation']['rppa']['scaler']
        
        # HiSeq缩放
        if hiseq_scaler_type == 'standard':
            self.hiseq_scaler = StandardScaler()
        elif hiseq_scaler_type == 'robust':
            self.hiseq_scaler = RobustScaler()
        elif hiseq_scaler_type == 'minmax':
            self.hiseq_scaler = MinMaxScaler()
            
        if self.hiseq_scaler:
            hiseq_features = self.hiseq_scaler.fit_transform(hiseq_features)
        
        # RPPA缩放
        if rppa_scaler_type == 'standard':
            self.rppa_scaler = StandardScaler()
        elif rppa_scaler_type == 'robust':
            self.rppa_scaler = RobustScaler()
        elif rppa_scaler_type == 'minmax':
            self.rppa_scaler = MinMaxScaler()
            
        if self.rppa_scaler:
            rppa_features = self.rppa_scaler.fit_transform(rppa_features)
        
        # 应用SMOTE处理类别不平衡（仅当启用时）
        if self.config['DATA_PROCESSING']['apply_smote']:
            logger.info("应用SMOTE处理类别不平衡...")
            try:
                smote = SMOTE(random_state=self.config['TRAINING_PARAMS']['random_seed'])
                # 为SMOTE合并特征
                combined_features = np.hstack((hiseq_features, rppa_features))
                combined_resampled, labels_resampled = smote.fit_resample(combined_features, labels)
                
                # 分离回两个模态
                hiseq_features = combined_resampled[:, :hiseq_features.shape[1]]
                rppa_features = combined_resampled[:, hiseq_features.shape[1]:]
                labels = labels_resampled
                
                logger.info(f"SMOTE后形状: HiSeq={hiseq_features.shape}, RPPA={rppa_features.shape}, 标签={labels.shape}")
            except Exception as e:
                logger.warning(f"SMOTE应用失败: {e}，使用原始数据")
        
        return hiseq_features, rppa_features, labels
    
    def get_dataloaders(self, hiseq_features, rppa_features, labels, test_size=0.2, batch_size=None):
        """
        创建训练和测试数据加载器
        
        参数:
            hiseq_features (numpy.ndarray): HiSeq特征
            rppa_features (numpy.ndarray): RPPA特征
            labels (numpy.ndarray): 标签
            test_size (float): 测试集比例
            batch_size (int): 批量大小
            
        返回:
            tuple: (train_loader, test_loader)
        """
        logger.info("创建数据加载器...")
        
        if batch_size is None:
            batch_size = self.config['TRAINING_PARAMS']['batch_size']
        
        # 分割训练集和测试集
        indices = np.arange(len(labels))
        train_indices, test_indices = train_test_split(
            indices, 
            test_size=test_size, 
            stratify=labels,
            random_state=self.config['TRAINING_PARAMS']['random_seed']
        )
        
        # 创建数据集
        train_dataset = MultiModalDataset(
            hiseq_features[train_indices],
            rppa_features[train_indices],
            labels[train_indices]
        )
        
        test_dataset = MultiModalDataset(
            hiseq_features[test_indices],
            rppa_features[test_indices],
            labels[test_indices]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False
        )
        
        logger.info(f"数据加载器创建完成。训练样本: {len(train_dataset)}, 测试样本: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def save_processed_data(self):
        """保存处理后的数据"""
        # 实现此方法保存处理后的数据
        pass 