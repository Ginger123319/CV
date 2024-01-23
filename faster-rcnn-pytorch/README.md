### Faster-RCNN

#### 代码权重下载

1. 本地代码路径：/home/jyf/code/faster-rcnn/faster-rcnn-pytorch
2. 本地权重路径：/home/jyf/code/faster-rcnn/faster-rcnn-pytorch/model_data/voc_weights_resnet.pth

#### 环境准备

1. cuda&cudnn：根据操作系统类型和显卡驱动版本安装对应的版本
2. anaconda：根据操作系统类型安装对应的版本；linux环境下推荐安装miniconda
3. 创建虚拟环境：conda create -n faster-rcnn python==3.7.4 -y
4. 激活虚拟环境：conda activate faster-rcnn
5. 进入requirements.txt文件所在位置：cd  /home/jyf/code/faster-rcnn/faster-rcnn-pytorch
6. 根据requirements.txt文件安装项目所需的包：pip install -r requirements.txt

#### 预测

1. 进入predict.py所在路径：cd  /home/jyf/code/faster-rcnn/faster-rcnn-pytorch
2. 执行预测文件：python predict.py --mode "dir_predict"
   1. 通过--mode参数指定预测的模式，目前支持"predict"、"dir_predict"两种模式
   2. 两种模式分别用于处理：单张图片预测、多张图片预测
3. 两种预测模式使用方法
   1. 单张图片预测：
      1. 在predict.py文件所在路径执行：python predict.py --mode "predict"
      2. 在光标位置输入单张图片的路径，预测成功后会继续提示输入图片，按ctrl+c可以退出预测
      3. 最终图片保存在项目中的img_out路径下，名称与输入图片名称一致
   2. 多张图片预测：
      1. 将需要预测的图片放到项目中的img目录下，或者将predict.py中的dir_origin_path修改为自定义的路径
      2. 在predict.py文件所在路径执行：python predict.py --mode "dir_predict"
      3. 最终图片保存在项目中的img_out路径下，名称与输入图片名称一致

#### 训练

1. 数据准备
   1. 需要VOC格式的数据进行训练
   2. 形如：<img src="C:\Users\Ginger\AppData\Local\Temp\企业微信截图_16793683576723.png" alt="img" style="zoom: 50%;" />
   3. 将图片放在 JPEGImages目录中，标签xml文件放在Annotation目录中
2. 数据处理
   1. 切到项目路径：cd /home/jyf/code/faster-rcnn/faster-rcnn-pytorch
   2. 在model_data中修改voc_classes.txt文件，将其中的类别名称修改为自定义名称
   3. 在voc_annotation.py文件中修改参数classes_path指向自定义数据集的类别文件(voc_classes.txt)
3. 参数调整
   1. 切到项目路径：cd /home/jyf/code/faster-rcnn/faster-rcnn-pytorch
   2. 在train.py文件中修改参数classes_path指向自定义数据集的类别文件(voc_classes.txt)
   3. 其他训练参数可以自定义修改，然后开启训练
   4. 训练后权重会保存在logs文件夹中
4. 结果预测
   1. 切到项目路径：cd /home/jyf/code/faster-rcnn/faster-rcnn-pytorch
   2. 在frcnn.py文件中
      1. 修改参数classes_path指向自定义数据集的类别文件(voc_classes.txt)
      2. 修改model_path指向在logs文件夹里训练好的权值文件
   3. 运行predict.py进行检测，具体检测方法参考--“**预测**”

#### 评估

1. 切到项目路径：cd /home/jyf/code/faster-rcnn/faster-rcnn-pytorch
2. 在get_map.py文件修改classes_path指向检测类别所对应的txt(voc_classes.txt)
3. 在frcnn.py文件修改model_path以及classes_path
   1. model_path指向logs文件夹里训练好的权值文件
   2. classes_path指向检测类别所对应的txt(voc_classes.txt)
4. 运行get_map.py即可获得评估结果，评估结果会保存在map_out文件夹中

