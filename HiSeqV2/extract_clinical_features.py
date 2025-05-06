import pandas as pd
import numpy as np

def extract_clinical_features(clinical_file):
    """
    从临床数据文件中提取特定特征
    
    Args:
        clinical_file: 临床数据文件路径
    
    Returns:
        pandas.DataFrame: 包含提取特征的数据框
    """
    # 读取临床数据文件
    df = pd.read_csv(clinical_file, sep='\t')
    
    # 提取需要的特征
    selected_features = [
        'sampleID',
        'days_to_death',
        'vital_status',
        'days_to_last_followup'
    ]
    
    # 创建结果数据框
    result_df = df[selected_features].copy()
    
    # 处理缺失值
    result_df['days_to_death'] = result_df['days_to_death'].fillna(-1)
    result_df['days_to_last_followup'] = result_df['days_to_last_followup'].fillna(-1)
    
    # 将生存状态转换为数值
    result_df['vital_status'] = result_df['vital_status'].map({'LIVING': 0, 'DECEASED': 1})
    
    print("提取的特征概况：")
    print("-" * 50)
    print(f"样本数量: {len(result_df)}")
    print("\n各特征的非空值数量：")
    print(result_df.count())
    print("\n数据预览：")
    print(result_df.head())
    
    return result_df

def save_features(df, output_file):
    """
    保存提取的特征到文件
    
    Args:
        df: 包含特征的数据框
        output_file: 输出文件路径
    """
    df.to_csv(output_file, index=False)
    print(f"\n特征数据已保存到: {output_file}")

def main():
    # 设置文件路径
    clinical_file = "TCGA.STAD.sampleMap_STAD_clinicalMatrix"
    output_file = "results/0/extracted_clinical_features.csv"
    
    # 提取特征
    features_df = extract_clinical_features(clinical_file)
    
    # 保存结果
    save_features(features_df, output_file)

if __name__ == "__main__":
    main()