import pandas as pd

# 读取TSV文件
df = pd.read_csv('./py/data_prossessing/HiSeqV2', sep='\t')

# 转置数据框
df_transposed = df.transpose()

# 将第一行设置为列名
df_transposed.columns = df_transposed.iloc[0]

# 删除第一行(现在它已经成为了列名)
df_transposed = df_transposed.drop('sample', axis=0)

# 保存为CSV文件
df_transposed.to_csv('./py/data_prossessing/HiSeqV2_transposed.csv')

print("数据已转置并保存为'HiSeqV2_transposed.csv'")