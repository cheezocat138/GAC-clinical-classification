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
    
    # 处理生存数据
    merged_df['event'] = merged_df['vital_status'].astype(float)
    
    # 计算实际生存天数 - 修复类型转换问题
    merged_df['survival_days'] = merged_df.apply(
        lambda row: float(row['days_to_death']) if row['vital_status'] == 1.0 
        else float(row['days_to_last_followup']) if pd.notna(row['days_to_last_followup']) and float(row['days_to_last_followup']) > 0 
        else 0, axis=1
    )
    
    # 过滤掉生存天数为0或负数的样本
    filtered_df = merged_df[merged_df['survival_days'] > 0]
    print(f"过滤后剩余样本: {len(filtered_df)}/{len(merged_df)}")
    
    # 保存处理后的数据
    filtered_df.to_csv(os.path.join(output_dir, 'survival_analysis_data.csv'), index=False)
    
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

def run_survival_analysis(rppa_file=DATA_PATHS['rppa_transposed'], 
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
    
    # 执行生存分析
    print("\n执行生存分析...")
    filtered_df, p_values = perform_survival_analysis(merged_df, output_dir)
    
    print(f"\n生存分析完成。详细结果已保存至: {output_dir}")
    
    return {
        'predictions_df': predictions_df, 
        'merged_df': merged_df,
        'filtered_df': filtered_df,
        'p_values': p_values
    }
