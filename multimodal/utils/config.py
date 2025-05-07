"""
配置模块

定义了多模态系统中使用的全局配置变量。
"""

import os
from pathlib import Path

# 项目根目录
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 数据路径配置
DATA_PATHS = {
    # 原始数据路径
    'raw_data_dir': ROOT_DIR / 'data' / 'raw',
    'processed_data_dir': ROOT_DIR / 'data' / 'processed',
    
    # 公共数据
    'survival_data': ROOT_DIR / 'data' / 'raw' / 'common' / 'survival_three_groups.csv',
    
    # HiSeq数据
    'hiseq_expression_data': ROOT_DIR / 'data' / 'raw' / 'hiseq' / 'HiSeqV2_transposed.csv',
    'hiseq_processed_data': ROOT_DIR / 'data' / 'processed' / 'hiseq_processed.csv',
    
    # RPPA数据
    'rppa_data': ROOT_DIR / 'data' / 'raw' / 'rppa' / 'RPPA_transposed.csv',
    'rppa_processed_data': ROOT_DIR / 'data' / 'processed' / 'rppa_processed.csv',
}

# 模型路径配置
MODEL_PATHS = {
    'model_dir': ROOT_DIR / 'models' / 'saved',
    'hiseq_model': ROOT_DIR / 'models' / 'saved' / 'hiseq_model.pth',
    'rppa_model': ROOT_DIR / 'models' / 'saved' / 'rppa_model.pth',
    'fusion_model': ROOT_DIR / 'models' / 'saved' / 'fusion_model.pth',
}

# 输出目录配置
OUTPUT_DIRS = {
    'results_dir': ROOT_DIR / 'results',
    'logs_dir': ROOT_DIR / 'logs',
    'plots_dir': ROOT_DIR / 'results' / 'plots',
    'model_dir': ROOT_DIR / 'models' / 'saved',
}

# 数据处理配置
DATA_PROCESSING = {
    # 是否应用数据平衡
    'apply_smote': True,
    
    # 特征选择配置
    'feature_selection': {
        'enabled': True,
        'max_features': 1000,  # 最大特征数
        'method': 'variance',  # 特征选择方法: 'variance', 'mutual_info', 'rfe'
    },
    
    # 数据转换配置
    'transformation': {
        'hiseq': {
            'scaler': 'robust',  # 'standard', 'robust', 'minmax', None
            'fill_na': 'median', # 'mean', 'median', 'constant', None
        },
        'rppa': {
            'scaler': 'robust',
            'fill_na': 'median',
        }
    },
}

# 训练配置
TRAINING_PARAMS = {
    'random_seed': 42,
    'batch_size': 32,
    'initial_lr': 0.001,
    'max_lr': 0.01,
    'min_lr': 0.0001,
    'n_splits': 5,     # 交叉验证折数
    'max_epochs': 150,
    'patience': 20,    # 早停轮数
    'class_weights': [1.0, 1.0, 1.0],  # 类别权重
}

# 可视化配置
PLOT_PARAMS = {
    'figsize_small': (8, 6),
    'figsize_medium': (12, 8),
    'figsize_large': (16, 12),
    'dpi': 300,
}

# 文件格式配置
FILE_FORMATS = {
    'model_fold': 'fold_{}_model.pth',
    'model_fold_acc': 'fold_{}_best_acc_model.pth',
    'cv_results': 'cv_results.png',
    'feature_importance': 'feature_importance.png',
}

# 统一配置字典
CONFIG = {
    'DATA_PATHS': DATA_PATHS,
    'MODEL_PATHS': MODEL_PATHS,
    'OUTPUT_DIRS': OUTPUT_DIRS,
    'DATA_PROCESSING': DATA_PROCESSING,
    'TRAINING_PARAMS': TRAINING_PARAMS,
    'PLOT_PARAMS': PLOT_PARAMS,
    'FILE_FORMATS': FILE_FORMATS,
}

# 确保必要的目录存在
def ensure_directories():
    """确保所有必要的目录存在"""
    dirs = [
        DATA_PATHS['raw_data_dir'] / 'common',
        DATA_PATHS['raw_data_dir'] / 'hiseq',
        DATA_PATHS['raw_data_dir'] / 'rppa',
        DATA_PATHS['processed_data_dir'],
        MODEL_PATHS['model_dir'],
        OUTPUT_DIRS['results_dir'],
        OUTPUT_DIRS['logs_dir'],
        OUTPUT_DIRS['plots_dir'],
    ]
    
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        
# 当直接运行此文件时，确保目录存在
if __name__ == "__main__":
    ensure_directories()
    print("配置加载完成，所有必要的目录已创建。") 