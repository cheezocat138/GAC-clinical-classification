"""
配置文件 - 存储所有文件路径和参数设置
"""
import os

# 基础路径
BASE_PATH = r'./'
RESULTS_PATH = os.path.join(BASE_PATH, 'results')

# 数据文件路径
DATA_PATHS = {
    'rppa_with_survival': os.path.join(RESULTS_PATH, '0', 'HiSeqV2_with_survival.csv'),
    'rppa_transposed': os.path.join(RESULTS_PATH, '0', 'HiSeqV2_transposed.csv'),
    'clinical': os.path.join(RESULTS_PATH, '0', 'extracted_clinical_features.csv'),
    'hiseq_with_survival': os.path.join(BASE_PATH, 'py', 'data_prossessing', 'HiSeqV2_with_survival.csv'),
    'hiseq_transposed': os.path.join(BASE_PATH, 'py', 'data_prossessing', 'HiSeqV2_transposed.csv'),
    'default_with_survival': os.path.join(BASE_PATH, 'py', 'data_prossessing', 'HiSeqV2_with_survival.csv'),
    'default_transposed': os.path.join(BASE_PATH, 'py', 'data_prossessing', 'HiSeqV2_transposed.csv'),
}

# 输出目录
OUTPUT_DIRS = {
    'final_model': os.path.join(RESULTS_PATH, 'final'),
    'survival_analysis': os.path.join(RESULTS_PATH, 'survival_analysis'),
}

# 模型文件路径
MODEL_PATHS = {
    'ensemble': os.path.join(OUTPUT_DIRS['final_model'], 'ensemble_survival_predictor_model.pth'),
    'final': os.path.join(OUTPUT_DIRS['final_model'], 'final_survival_predictor_model.pth'),
}

# 预测结果文件路径
PREDICTION_PATH = os.path.join(OUTPUT_DIRS['survival_analysis'], 'predictions.csv')

# 训练参数
TRAINING_PARAMS = {
    'random_seed': 42,
    'n_splits': 5,
    'batch_size': 32,
    'max_epochs': 150,
    'initial_lr': 0.01,
    'min_lr': 0.0001,
    'max_lr': 0.01,
    'patience': 30,
    'class_weights': [1.0, 1.0, 1.0],
}

# 特征选择参数
FEATURE_SELECTION_PARAMS = {
    'mi_k': 500,
    'f_k': 500,
    'rfe_k': 300,
    'min_features': 300,
}

# 聚类参数
CLUSTER_PARAMS = {
    'n_clusters': 3,
    'random_state': 42,
}

# 文件格式
FILE_FORMATS = {
    'model_fold': 'best_model_fold{}.pth',
    'model_fold_acc': 'best_acc_model_fold{}.pth',
    'confusion_matrix': 'confusion_matrix.png',
    'cv_results': 'cv_training_results.png',
    'feature_importance': 'feature_importance.png',
}

# 其他常量
PLOT_PARAMS = {
    'dpi': 300,
    'figsize_large': (24, 16),
    'figsize_medium': (12, 10),
    'figsize_small': (10, 8),
    'group_names': ['Group 1', 'Group 2', 'Group 3'],
    'colors': ['blue', 'red', 'green', 'purple', 'orange'],
}
