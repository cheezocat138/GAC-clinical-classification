"""
工具模块

该模块包含多模态系统使用的工具类和函数。
主要功能包括：
1. 配置管理
2. 日志记录
3. 通用辅助函数
"""

# 使用绝对导入
from multimodal.utils.config import CONFIG, DATA_PATHS, MODEL_PATHS, OUTPUT_DIRS, DATA_PROCESSING, TRAINING_PARAMS, ensure_directories
from multimodal.utils.logger import LOGGER, get_logger

__all__ = [
    'CONFIG',
    'DATA_PATHS',
    'MODEL_PATHS',
    'OUTPUT_DIRS',
    'DATA_PROCESSING',
    'TRAINING_PARAMS',
    'ensure_directories',
    'LOGGER',
    'get_logger',
] 