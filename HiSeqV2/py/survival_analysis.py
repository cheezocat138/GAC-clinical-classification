import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from predict import predict_from_rppa, load_model
from utils import ensure_dir
from config import DATA_PATHS, MODEL_PATHS, OUTPUT_DIRS, PREDICTION_PATH, PLOT_PARAMS

def merge_predictions_with_clinical(prediction_df, clinical_file):
    """将预测结果与临床数据合并"""
    # 加载临床数据
    clinical_data = pd.read_csv(clinical_file)
    
    # 处理样本ID格式，去掉'-01'后缀以便于匹配
    prediction_df['sampleID'] = prediction_df['sample'].apply(
        lambda x: x.split('-01')[0] if '-01' in x else x
    )
    
    clinical_data['sampleID_short'] = clinical_data['sampleID'].apply(
        lambda x: x.split('-01')[0] if '-01' in x else x
    )
    
    # 合并数据
    merged_df = pd.merge(
        clinical_data, prediction_df,
        left_on='sampleID_short', right_on='sampleID',
        how='inner'
    )
    
    # 清理合并后的数据框
    merged_df = merged_df.drop(['sampleID_y', 'sample'], axis=1)
    merged_df = merged_df.rename(columns={'sampleID_x': 'sampleID'})
    
    return merged_df

def perform_survival_analysis(merged_df, output_dir=OUTPUT_DIRS['survival_analysis']):
    """进行生存分析并绘制Kaplan-Meier曲线"""
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    # 打印数据检查信息
    print("进行生存分析前数据检查:")
    print(f"总样本数: {len(merged_df)}")
    print(f"包含NaN的行数: {merged_df.isna().any(axis=1).sum()}")
    
    # 确保关键列存在
    required_columns = ['vital_status', 'days_to_death', 'days_to_last_followup', 'predicted_group']
    for col in required_columns:
        if col not in merged_df.columns:
            raise ValueError(f"缺少关键列 '{col}'")
    
    # 打印各列数据类型
    print("列数据类型:")
    print(merged_df[required_columns].dtypes)
    
    # 处理生存数据
    # 确保vital_status是数值类型
    merged_df['event'] = merged_df['vital_status'].astype(float)
    
    # 计算实际生存天数 - 修复类型转换问题
    print("计算生存天数...")
    
    # 安全地转换生存时间数据
    merged_df['days_to_death'] = pd.to_numeric(merged_df['days_to_death'], errors='coerce')
    merged_df['days_to_last_followup'] = pd.to_numeric(merged_df['days_to_last_followup'], errors='coerce')
    
    # 应用生存天数计算逻辑
    merged_df['survival_days'] = merged_df.apply(
        lambda row: float(row['days_to_death']) if pd.notna(row['days_to_death']) and row['vital_status'] == 1.0 
        else float(row['days_to_last_followup']) if pd.notna(row['days_to_last_followup']) and float(row['days_to_last_followup']) > 0 
        else 0, axis=1
    )
    
    # 检查计算后的生存天数
    print(f"生存天数统计: 最小值={merged_df['survival_days'].min()}, 最大值={merged_df['survival_days'].max()}, 均值={merged_df['survival_days'].mean():.2f}")
    print(f"生存天数为0的样本数: {(merged_df['survival_days'] == 0).sum()}")
    
    # 过滤掉生存天数为0或负数的样本
    filtered_df = merged_df[merged_df['survival_days'] > 0]
    print(f"过滤后剩余样本: {len(filtered_df)}/{len(merged_df)}")
    
    # 保存处理后的数据
    filtered_df.to_csv(os.path.join(output_dir, 'survival_analysis_data.csv'), index=False)
    
    # 检查每个组的样本数量
    for group in sorted(filtered_df['predicted_group'].unique()):
        group_count = len(filtered_df[filtered_df['predicted_group'] == group])
        print(f"Group {group}: {group_count} samples")
    
    # 创建Kaplan-Meier生存曲线
    kmf = KaplanMeierFitter()
    groups = sorted(filtered_df['predicted_group'].unique())
    
    plt.figure(figsize=PLOT_PARAMS['figsize_small'])
    colors = PLOT_PARAMS['colors']
    
    p_values = []
    
    # 绘制每个组的生存曲线
    for i, group in enumerate(groups):
        group_data = filtered_df[filtered_df['predicted_group'] == group]
        print(f"Group {group}: {len(group_data)} samples")
        kmf.fit(group_data['survival_days'], group_data['event'], label=f'Group {group}')
        kmf.plot(ci_show=True, color=colors[i % len(colors)])
        
        # 计算组间两两比较的log-rank检验
        for j, compare_group in enumerate(groups):
            if j > i:  # 避免重复比较
                compare_data = filtered_df[filtered_df['predicted_group'] == compare_group]
                if len(group_data) > 0 and len(compare_data) > 0:
                    results = logrank_test(
                        group_data['survival_days'], compare_data['survival_days'], 
                        group_data['event'], compare_data['event']
                    )
                    p_values.append((group, compare_group, results.p_value))
    
    # 设置图表属性
    plt.title('Kaplan-Meier Survival Curves by Predicted Group', fontsize=16)
    plt.xlabel('Time (Days)', fontsize=14)
    plt.ylabel('Survival Probability', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # 保存图表
    plt.savefig(os.path.join(output_dir, 'kaplan_meier_curves.png'), dpi=PLOT_PARAMS['dpi'], bbox_inches='tight')
    plt.close()
    
    # 将Log-rank检验结果保存到文件
    logrank_results = pd.DataFrame(p_values, columns=['Group 1', 'Group 2', 'p-value'])
    logrank_results['Significant'] = logrank_results['p-value'] < 0.05
    logrank_results.to_csv(os.path.join(output_dir, 'logrank_test_results.csv'), index=False)
    
    # 打印Log-rank检验结果
    print("\nLog-rank检验结果 (p-value):")
    for g1, g2, p in p_values:
        print(f"Group {g1} vs Group {g2}: p = {p:.6f}" + 
              (" (显著差异)" if p < 0.05 else ""))
    
    return filtered_df, p_values

def run_survival_analysis(rppa_file=DATA_PATHS['default_transposed'], 
                         clinical_file=DATA_PATHS['clinical'], 
                         model_path=MODEL_PATHS['ensemble'], 
                         output_dir=OUTPUT_DIRS['survival_analysis']):
    """运行完整的生存分析流程"""
    # 确保输出目录存在
    ensure_dir(output_dir)
    
    print("开始运行生存组预测与生存分析...")
    
    # 从RPPA数据预测生存组别
    print("预测样本生存组别...")
    prediction_file = os.path.join(output_dir, 'predictions.csv')
    predictions_df = predict_from_rppa(rppa_file, model_path, prediction_file)
    
    # 统计预测组别分布
    groups = predictions_df['predicted_group'].value_counts().sort_index()
    print("\n预测生存组别分布:")
    for group, count in groups.items():
        print(f"Group {group}: {count} samples ({count/len(predictions_df)*100:.1f}%)")
    
    # 合并预测结果与临床数据
    print("\n合并预测结果和临床数据...")
    merged_df = merge_predictions_with_clinical(predictions_df, clinical_file)
    print(f"成功合并 {len(merged_df)} 个有效样本")
    
    # 类型转换修正 - 确保数值列是数值类型
    # 尝试将survival相关列转换为数值类型
    numeric_columns = ['days_to_death', 'days_to_last_followup', 'vital_status']
    for col in numeric_columns:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
    
    # 保存合并后的数据
    merged_file = os.path.join(output_dir, 'merged_clinical_with_predictions.csv')
    merged_df.to_csv(merged_file, index=False)
    print(f"合并数据已保存至: {merged_file}")
    
    # 检查并处理NaN值
    nan_rows = merged_df.isna().any(axis=1).sum()
    if nan_rows > 0:
        print(f"\n检测到{nan_rows}行包含NaN值，正在清理数据...")
        # 删除包含NaN的行
        merged_df = merged_df.dropna()
        cleaned_file = os.path.join(output_dir, 'merged_clinical_with_predictions_cleaned.csv')
        merged_df.to_csv(cleaned_file, index=False)
        print(f"清理后的数据已保存至: {cleaned_file}")
    
    # 执行生存分析
    print("\n执行生存分析...")
    try:
        filtered_df, p_values = perform_survival_analysis(merged_df, output_dir)
        print(f"\n生存分析完成。详细结果已保存至: {output_dir}")
        
        return {
            'predictions_df': predictions_df, 
            'merged_df': merged_df,
            'filtered_df': filtered_df,
            'p_values': p_values
        }
    except Exception as e:
        print(f"生存分析过程中出错: {str(e)}")
        # 返回部分结果
        return {
            'predictions_df': predictions_df,
            'merged_df': merged_df,
            'error': str(e)
        }
