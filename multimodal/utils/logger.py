"""
日志模块

提供统一的日志记录功能。
"""

import os
import sys
import logging
from pathlib import Path
import datetime

# 使用绝对导入
from multimodal.utils.config import OUTPUT_DIRS

def setup_logger(name, log_file=None, level=logging.INFO):
    """设置日志记录器
    
    Args:
        name: 日志记录器名称
        log_file: 日志文件路径，如果为None则使用默认路径
        level: 日志级别
        
    Returns:
        配置好的日志记录器
    """
    # 创建日志记录器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 如果已有处理器，不再添加
    if logger.handlers:
        return logger
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # 设置格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(console_handler)
    
    # 如果指定了日志文件，添加文件处理器
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name, module_name=None):
    """获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称
        module_name: 模块名称，用于生成日志文件名
        
    Returns:
        日志记录器
    """
    # 确保日志目录存在
    os.makedirs(OUTPUT_DIRS['logs_dir'], exist_ok=True)
    
    # 生成日志文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if module_name:
        log_filename = f"{module_name}_{timestamp}.log"
    else:
        log_filename = f"{name}_{timestamp}.log"
    
    log_path = OUTPUT_DIRS['logs_dir'] / log_filename
    
    # 设置并返回日志记录器
    return setup_logger(name, log_path)

# 默认主日志记录器
LOGGER = get_logger('multimodal') 