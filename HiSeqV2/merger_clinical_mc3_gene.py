import pandas as pd

# 读取两个数据集
mc3_gene_df = pd.read_csv('./py/data_prossessing/HiSeqV2_transposed.csv')
survival_df = pd.read_csv('./survival_three_groups.csv')

# 检查RPPA数据的索引/样本ID列名
# 如果第一列是样本ID但没有列名，则设置列名
if mc3_gene_df.columns[0] == "Unnamed: 0":
    mc3_gene_df = pd.read_csv('./py/data_prossessing/HiSeqV2_transposed.csv', index_col=0)
    # 将索引列转换为常规列便于合并
    mc3_gene_df.reset_index(inplace=True)
    mc3_gene_df.rename(columns={'index': 'sampleID'}, inplace=True)

# 确保survival_df中的sampleID列与RPPA数据的ID列匹配格式
# TCGA数据通常有"-01"或"-11"后缀，这里统一保留主要ID部分以便匹配
survival_df['match_id'] = survival_df['sampleID'].str.split('-').str[:-1].str.join('-')
survival_unique = survival_df.drop_duplicates(subset='match_id')[['match_id', 'survival_group_code']]

# 如果mc3_gene_df中的样本ID也包含后缀，创建匹配ID列
if 'sampleID' in mc3_gene_df.columns:
    mc3_gene_df['match_id'] = mc3_gene_df['sampleID'].str.split('-').str[:-1].str.join('-')
else:
    # 假设第一列是样本ID
    first_col_name = mc3_gene_df.columns[0]
    mc3_gene_df['match_id'] = mc3_gene_df[first_col_name].str.split('-').str[:-1].str.join('-')

# 基于match_id合并数据集
merged_df = pd.merge(mc3_gene_df, survival_unique, on='match_id', how='left')

# 记录合并前的行数
total_before_filter = len(merged_df)

# 去除survival_group_code为空的行
merged_df = merged_df.dropna(subset=['survival_group_code'])

# 删除临时匹配ID列
merged_df.drop('match_id', axis=1, inplace=True)

# 保存合并后的数据集
output_path = './py/data_prossessing/HiSeqV2_with_survival.csv'
merged_df.to_csv(output_path, index=False)

print(f"合并完成，已保存至 {output_path}")
print(f"合并前: HiSeqV2数据形状 {mc3_gene_df.shape}, 生存数据形状 {survival_df.shape}")
print(f"合并后数据形状: {merged_df.shape}")

# 计算有多少样本成功匹配到了生存分类信息
matched = merged_df.shape[0]  # 已经过滤掉了空值，所有行都有生存分类信息
print(f"成功匹配到生存分类信息的样本数: {matched}")
print(f"过滤前总样本数: {total_before_filter}, 过滤掉的样本数: {total_before_filter - matched}")