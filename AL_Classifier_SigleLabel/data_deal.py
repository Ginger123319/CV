import ast
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CSVDataset(Dataset):
    def __init__(self, label_path, unlabeled_path, input_shape, is_train=True):
        self.transform = []
        self.transform.append(transforms.ToTensor())
        if is_train:
            self.transform.append(transforms.RandomHorizontalFlip(0.5))
        self.transform = transforms.Compose(self.transform)

        self.data = []
        self.class_name = []
        self.input_shape = input_shape
        df = pd.read_csv(label_path)

        df.loc[:, 'label'] = [ast.literal_eval(label)['annotations'][0]['category_id'] for label in df.loc[:, 'label']]
        # print(df)
        from sklearn.model_selection import train_test_split
        df_train, df_valid = train_test_split(df, test_size=0.2, random_state=1, shuffle=True,
                                              stratify=df["label"])
        print(df_valid.reset_index(drop=True))
        exit()
        # 统计类别名称
        for label in df['label']:
            label = ast.literal_eval(label)
            self.class_name.append(label['annotations'][0]['category_id'])
        self.class_name = list(set(self.class_name))
        self.class_name.sort()

        # 构建数据集
        if not is_train:
            df = pd.read_csv(unlabeled_path)
        # 对于多标签分类对标签的处理
        for index, row in df.iterrows():
            tag = np.zeros(len(self.class_name), dtype=np.float32)
            if not pd.isna(row['label']):
                tag_list = [self.class_name.index(cat['category_id']) for cat in
                            ast.literal_eval(row['label'])['annotations']]
                tag[tag_list] = 1
                print(tag)
                exit()
                tag = self.class_name.index(ast.literal_eval(row['label'])['annotations'][0]['category_id'])
            else:
                # 如果没有标签，设置一个为1的假标签
                tag = 1
            self.data.append((row['id'], row['path'], tag))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample_id = self.data[index][0]
        img_path = self.data[index][1]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.input_shape)
        tag = self.data[index][2]
        return sample_id, self.transform(img), np.int64(tag), index

    def parse_label(self, target):
        pass


if __name__ == '__main__':
    # 要求输入的标注数据中标签列不能为空，且必须有类别信息
    input_label_path = 'temp.csv'
    input_unlabeled_path = 'empty.csv'
    data = CSVDataset(input_label_path, input_unlabeled_path, input_shape=(256, 256), is_train=True)
    dataLoader = DataLoader(data, batch_size=4, shuffle=False)
    for i, (id, x, y, idx) in enumerate(dataLoader):
        # print(i)
        print(x.shape)
        print(y.shape)
        print(idx)
        exit()
