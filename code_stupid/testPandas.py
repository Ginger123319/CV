import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

path = r"D:\Python\test_jzyj\images"
X, y = np.arange(10).reshape((5, 2)), range(5)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

print(X_train)

print(y_train)

print(X_test)

print(y_test)

train_test_split(y, shuffle=False)

d = {'col1': [1, 2], 'col2': [3, 4]}
# df = pd.DataFrame({'col1': 1, 'col2': 3}, index=[0])
df = pd.DataFrame(data=d)
print(df)
exit()


print(df.dtypes.items())
for k, v in df.dtypes.items():
    print(k, v)
    if v.name == "int64":
        print(type(df[k]))
        print(type(df.loc[:, k]))
        df.loc[:, k] = df[k].astype(float)
print(df.dtypes)
print(df['col2'].values)

df.dropna(axis=0, inplace=True)
print(df.dropna(axis=0, inplace=True))

print(isinstance(df, pd.DataFrame))
print(np.unique(df['col2']))
print(df['col1'].shape)
df_train = df.copy()
print(df_train == df)
print(df_train)
df_train['col3'] = df['col2']
print(df['col2'])
print(df_train)
print(df)

csv_path = os.path.join(path, "csv")
# os.mkdir(csv_path)
df_train.to_csv(csv_path)
for col1, col2 in zip(df['col1'], df['col2']):
    print(col1, col2)
print(df_train.take([0]))
img_dir = r"D:\Python\test_jzyj\images\test"
_, _, files = next(os.walk(img_dir))
print(files)
# for i in files:
# 如果是多对1，1的那一列会复制同样的值进行填充;否则All arrays must be of the same length
df_new = pd.DataFrame({"A": files, "B": [2, 3, None, 3, 3, 3, ]})
df_new2 = pd.DataFrame({"A": files, "B": [2, 5, None, 8, 1, 3, ]})
print(df_new)
df_new3 = np.arange(1, 15, 1, dtype=int).reshape(7, 2).view(np.float32)

print(type(df_new3))
print(df_new3)
new_df = pd.concat([df_new, pd.DataFrame(df_new3, columns=["prob_{}".format(c) for c in ('cat', 'dog')])], axis=1)

print(new_df)
# 区别在于是否保留列名
print(new_df["A"])
print(new_df[["A"]])

print(new_df.reset_index(drop=True))
print(new_df)


def remove_file(dir_path, file_pattern):
    from pathlib import Path
    # 递归查找匹配pattern的文件（包括目录），保存再g中
    g = Path(dir_path).rglob(file_pattern)

    print(type(g))
    # <class 'generator'>
    # 将g中的所有文件都删除，
    for f in g:
        print(type(f))
        if f.is_file():
            print("Removing tmp file... {}".format(f.absolute()))
            f.unlink()


remove_file(path, "*.txt")


# 把float16转换为float类型
def filter_float16(df_16):
    for k1, v1 in df_16.dtypes.items():
        if v1.name == "float16":
            df_16[k1] = df_16[k1].astype(float)
    return df_16
