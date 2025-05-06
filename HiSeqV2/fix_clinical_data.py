import pandas as pd
import numpy as np

# 读取合并后的临床数据文件
print("读取合并后的临床数据文件...")
merged_file = './results/survival_analysis/merged_clinical_with_predictions.csv'
df = pd.read_csv(merged_file)

# 检查NaN值
print("原始数据形状:", df.shape)
print("包含NaN的行数:", df.isna().any(axis=1).sum())
print("每列NaN值数量:")
print(df.isna().sum())

# 显示包含NaN的行
nan_rows = df[df.isna().any(axis=1)]
print("\n包含NaN值的行:")
print(nan_rows)

# 处理方法1：删除包含NaN的行
df_cleaned = df.dropna()
print("\n删除NaN后的数据形状:", df_cleaned.shape)

# 保存清理后的数据
cleaned_file = './results/survival_analysis/merged_clinical_with_predictions_cleaned.csv'
df_cleaned.to_csv(cleaned_file, index=False)
print(f"清理后的数据已保存到: {cleaned_file}")

# 或者处理方法2：填充NaN值（如果需要）
# 这里我们为vital_status填充0（假设0表示死亡，1表示存活）
# 为days_to_death填充该组的平均值
df_filled = df.copy()

# 获取预测组
group1_mask = df['predicted_group'] == 1
group2_mask = df['predicted_group'] == 2 
group3_mask = df['predicted_group'] == 3

# 为每个组计算days_to_death的平均值（排除NaN）
mean_days_group1 = df.loc[group1_mask, 'days_to_death'].mean()
mean_days_group2 = df.loc[group2_mask, 'days_to_death'].mean()
mean_days_group3 = df.loc[group3_mask, 'days_to_death'].mean()

# 根据预测组填充days_to_death的NaN值
for idx, row in df_filled.loc[df_filled['days_to_death'].isna()].iterrows():
    if row['predicted_group'] == 1:
        df_filled.at[idx, 'days_to_death'] = mean_days_group1
    elif row['predicted_group'] == 2:
        df_filled.at[idx, 'days_to_death'] = mean_days_group2
    else:
        df_filled.at[idx, 'days_to_death'] = mean_days_group3

# 填充vital_status（假设缺失为0，表示死亡）
df_filled['vital_status'] = df_filled['vital_status'].fillna(0)

filled_file = './results/survival_analysis/merged_clinical_with_predictions_filled.csv'
df_filled.to_csv(filled_file, index=False)
print(f"填充NaN后的数据已保存到: {filled_file}")

print("\n建议使用清理后的数据（删除NaN行）进行生存分析，这样结果更可靠") 