import pandas as pd
import numpy as np
from pathlib import Path

class HiSeqDataProcessor:
    def __init__(self, data_dir: str):
        """
        初始化HiSeq数据处理器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        
    def load_and_process_data(self, expression_file: str, survival_file: str = 'survival_three_groups.csv') -> pd.DataFrame:
        """
        加载并处理基因表达数据
        
        Args:
            expression_file: 基因表达数据文件名
            survival_file: 生存数据文件名
            
        Returns:
            处理后的数据框，包含基因表达数据和生存组信息
        """
        # 读取基因表达数据
        expression_data = pd.read_csv(self.data_dir / expression_file, index_col=0)
        
        # 读取生存数据
        survival_data = pd.read_csv(self.data_dir / survival_file)
        
        # 确保样本ID匹配
        common_samples = set(expression_data.index) & set(survival_data['sample_id'])
        
        # 提取共同样本的数据
        expression_filtered = expression_data.loc[common_samples]
        survival_filtered = survival_data[survival_data['sample_id'].isin(common_samples)]
        
        # 将生存组信息添加到表达数据中
        survival_dict = dict(zip(survival_filtered['sample_id'], survival_filtered['survival_group_code']))
        expression_filtered['survival_group_code'] = expression_filtered.index.map(survival_dict)
        
        return expression_filtered
    
    def save_processed_data(self, data: pd.DataFrame, output_file: str):
        """
        保存处理后的数据
        
        Args:
            data: 处理后的数据框
            output_file: 输出文件名
        """
        data.to_csv(self.data_dir / output_file) 