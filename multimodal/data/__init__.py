"""
数据模块

该模块包含用于处理RPPA和HiSeq数据的类和函数。
主要功能包括：
1. 数据加载和预处理
2. 特征工程
3. 数据转换和标准化
4. 数据验证和清洗
5. 多模态数据整合
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from imblearn.over_sampling import SMOTE

# 使用绝对导入
from multimodal.data.rppa_data import RPPADataProcessor
from multimodal.data.hiseq_data import HiSeqDataProcessor
from multimodal.data.multimodal_dataset import MultiModalDataProcessor, MultiModalDataset

__all__ = [
    'RPPADataProcessor',
    'HiSeqDataProcessor',
    'MultiModalDataProcessor',
    'MultiModalDataset',
]

# 版本信息
__version__ = '0.1.0'