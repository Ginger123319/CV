import pandas as pd

# 创建示例DataFrame
df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
df2 = pd.DataFrame({'A': [1, 2, 4], 'C': [7, 8, 9]})

# 按照df1和df2中的'A'列的值是否相同来合并DataFrame
merged_df = pd.merge(df1, df2, on='A', how='outer')

# 输出结果
print(merged_df)
