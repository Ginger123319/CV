import pandas as pd
import numpy as np

x = np.clip(np.array([1, 2]), 2, 3)

if None:
    print(not [])
content = {'image_col': [], 'label_col': []}
df = pd.DataFrame(content)
# df = df[~df['label_col'].apply(lambda x: [] == x)]
image_list = []
class_name = {}
file_name = []
for img_path, labels in zip(df['image_col'], df['label_col']):
    # img = read_rgb_img(img_path)
    print(111111111111)
    if isinstance(img, str):
        continue

    file_name.append(img_path)

    label_list = []
    for label in labels:
        if label_type == "103":
            cls = label[-1]
        else:
            cls = label
        if cls not in class_name.keys():
            class_name[cls] = len(class_name) + 1
        if label_type == "103":
            label_list.append(label[:4] + [class_name[cls]])
        else:
            label_list.append(class_name[cls])
    image_list.append([img, label_list])
print(df.empty)

if not False and not df.empty:
    print(22222222)
