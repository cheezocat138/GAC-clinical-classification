import pandas as pd
import numpy as np
from pathlib import Path
import os

from multimodal.utils.config import CONFIG, DATA_PATHS, DATA_PROCESSING

class HiSeqDataProcessor:
    def __init__(self, config=None):
        """
        初始化HiSeq数据处理器
        
        Args:
            config: 配置字典，如果为None则使用默认配置
        """
        self.config = config if config is not None else CONFIG
        # 设置数据目录为项目根目录
        self.data_dir = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    def load_data(self):
        """
        加载数据的简化接口，内部调用load_and_process_data
        
        Returns:
            pandas.DataFrame: 处理后的数据框
        """
        return self.load_and_process_data()
        
    def load_and_process_data(self, expression_file: str = None, 
                             survival_file: str = None) -> pd.DataFrame:
        """
        加载并处理基因表达数据
        
        Args:
            expression_file: 基因表达数据文件路径，如果为None则使用配置中的路径
            survival_file: 生存数据文件路径，如果为None则使用配置中的路径
            
        Returns:
            处理后的数据框，包含基因表达数据和生存组信息
        """
        # 使用配置中的文件路径，如果未指定
        if expression_file is None:
            expression_path = self.config['DATA_PATHS']['hiseq_expression_data']
        else:
            expression_path = self.data_dir / expression_file
            
        if survival_file is None:
            survival_path = self.config['DATA_PATHS']['survival_data']
        else:
            survival_path = self.data_dir / survival_file
        
        # 读取基因表达数据
        print(f"加载基因表达数据：{expression_path}")
        expression_data = pd.read_csv(expression_path, index_col=0)
        
        # 读取生存数据
        print(f"加载生存数据：{survival_path}")
        survival_data = pd.read_csv(survival_path)
        
        # 调整样本ID列名
        if 'sampleID' in survival_data.columns:
            sample_id_col = 'sampleID'
        else:
            sample_id_col = 'sample_id'
        
        # 确保样本ID匹配
        print(f"基因表达样本数量：{len(expression_data.index)}")
        print(f"生存数据样本数量：{len(survival_data[sample_id_col])}")
        
        common_samples = set(expression_data.index) & set(survival_data[sample_id_col])
        print(f"匹配样本数量：{len(common_samples)}")
        
        # 提取共同样本的数据
        expression_filtered = expression_data.loc[list(common_samples)]
        survival_filtered = survival_data[survival_data[sample_id_col].isin(common_samples)]
        
        # 将生存组信息添加到表达数据中
        survival_dict = dict(zip(survival_filtered[sample_id_col], survival_filtered['survival_group_code']))
        expression_filtered['survival_group_code'] = expression_filtered.index.map(survival_dict)
        
        # 检查是否有缺失值
        if expression_filtered['survival_group_code'].isna().any():
            print("警告：部分样本缺失生存组信息，这些样本将被移除")
            expression_filtered = expression_filtered.dropna(subset=['survival_group_code'])
            
        # 将survival_group_code转换为整数
        expression_filtered['survival_group_code'] = expression_filtered['survival_group_code'].astype(int)
        
        # 应用数据转换配置
        transform_config = self.config['DATA_PROCESSING']['transformation']['hiseq']
        
        # 填充缺失值
        if transform_config['fill_na'] is not None:
            numeric_cols = expression_filtered.select_dtypes(include=[np.number]).columns
            
            if transform_config['fill_na'] == 'mean':
                fill_values = expression_filtered[numeric_cols].mean()
            elif transform_config['fill_na'] == 'median':
                fill_values = expression_filtered[numeric_cols].median()
            elif transform_config['fill_na'] == 'constant':
                fill_values = 0
            else:
                fill_values = expression_filtered[numeric_cols].median()
                
            expression_filtered[numeric_cols] = expression_filtered[numeric_cols].fillna(fill_values)
            print(f"使用{transform_config['fill_na']}方法填充缺失值")
            
        return expression_filtered
    
    def save_processed_data(self, data: pd.DataFrame, output_file: str = None):
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据框
            output_file: 输出文件路径，如果为None则使用配置中的路径
        """
        if output_file is None:
            output_path = self.config['DATA_PATHS']['hiseq_processed_data']
        else:
            output_path = self.data_dir / output_file
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        print(f"保存处理后的数据到：{output_path}")
        print(f"处理后的数据维度：{data.shape}")
        data.to_csv(output_path)
        print("保存完成")
        
    def preprocess(self, apply_feature_selection: bool = None):
        """
        执行完整的预处理流程
        
        Args:
            apply_feature_selection: 是否应用特征选择，如果为None则使用配置中的设置
        """
        print("开始HiSeq数据预处理...")
        
        # 加载并处理数据
        processed_data = self.load_and_process_data()
        
        # 应用特征选择
        if apply_feature_selection is None:
            apply_feature_selection = self.config['DATA_PROCESSING']['feature_selection']['enabled']
            
        if apply_feature_selection:
            # 特征选择代码将在这里实现
            print("应用特征选择...")
            # processed_data = self._apply_feature_selection(processed_data)
            
        # 保存处理后的数据
        self.save_processed_data(processed_data)
        
        print("HiSeq数据预处理完成")
        return processed_data 