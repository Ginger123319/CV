### 图像分类-Efficientnet_b7

#### 代码权重下载

1. 本地代码路径：D:/Python/code/jz_efficientnet_b7
2. 初始权重路径：D:/Python/pretrained_weights/checkpoints/efficientnet_b7_lukemelas-dcc49843.pth

1. cuda&cudnn：根据操作系统类型和显卡驱动版本安装对应的版本
2. anaconda：根据操作系统类型安装对应的版本；linux环境下推荐安装miniconda
3. 创建虚拟环境：conda create -n efficientnet python==3.7.4 -y
4. 激活虚拟环境：conda activate efficientnet
5. 进入requirements.txt文件所在位置：D:/Python/code/jz_efficientnet_b7
6. 根据requirements.txt文件安装项目所需的包：pip install -r requirements.txt

#### 训练

1. 数据准备

   1. 目前支持按目录分类、voc格式、coco格式的图像分类数据集

   2. 使用该算子，需要将数据集转为csv格式的文件再执行训练预测过程

   3. 转换函数所在目录：D:/Python/code/jz_efficientnet_b7/data2csv/img2csv.py

   4. 修改该py文件中的几个参数：

      ```
      # 1.修改参数is_train来修改转换模式
      #   训练模式，生成csv中有标签；测试模式，生成的csv中没有标签
      # 2.修改data_type指定数据集的类型
      # 3.修改对应的数据集路径为自定义路径即可
      ```

2. 参数调整

   1. 切到项目路径：D:/Python/code/jz_efficientnet_b7

   2. 在main.py文件中修改

      ```
      # 修改此处控制是否只进行预测
      only_predict = False
      # 修改此处可以改变验证集比例
      val_size = 0.2
      # 修改此处修改初始权重位置
      weights = r'D:/Python/pretrained_weights/checkpoints/efficientnet_b7_lukemelas-dcc49843.pth'
      ```

   3. 其他训练参数可以自定义修改，然后开启训练(执行main.py)

      ```
      Config = {
          "class_num": class_num,
          "optimizer": 'Adam',
          'fit_batch_size': 32,
          "FIT_epochs": 30,
          "class_name": class_name,
          "input_shape": 224,
          "image_col": "path",
          "label_col": label_col,
          "val_ratio": val_size,
          "id_col": 'id',
          "weights_path": weights_path
      }
      ```

   4. 训练后权重（best_network.pth）、模型（model.pkl）以及分类的类别名称（class.pt）会保存在exp_dir/output文件夹中；

      1. 全路径为(D:/Python/code/jz_efficientnet_b7/exp_dir/output)

#### 预测

1. 在train.py文件中将only_predict修改only_predict = True
2. 执行main.py
3. 最终结果保存在exp_dir/output/output_query.csv中；label列即为预测的类别

#### 评估

1. 在训练过程中会打印训练后在验证集上的最佳精度，以此精度作为评估指标

