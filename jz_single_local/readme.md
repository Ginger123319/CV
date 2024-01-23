
# Table of Contents

1.  [智能标注模型开发文档](#orgdca0b4e)
    1.  [模型侧功能描述](#org0493d5f)
    2.  [如何新增模型用于智能标注](#orgf435422)
    3.  [生成新模型的流程](#org3c83abb)
        1.  [准备数据文件](#org20de915)
        2.  [在 `create_pipeline_demo.py` 中修改参数](#org176c3b4)
        3.  [在demo目录中开发模型](#org925253f)
        4.  [启动create\_pipeline\_demo.py生成模型文件](#org4aaccb0)


<a id="orgdca0b4e"></a>

# 智能标注模型开发文档


<a id="org0493d5f"></a>

## 模型侧功能描述

1.  针对需要智能标注的数据集，先做初始化的标注，每种场景的初始标注样本条数不同。
2.  服务端将初始化标注的数据转成input\_label.csv, 后续新增难例标注都放在input\_added\_label.csv里, 未标注的数据放在input\_unlabel.csv
3.  从模型仓库提供用于智能标注的模型
4.  将标注数据、未标注数据、模型传给智能标注算子
5.  执行智能标注算子：
    1.  拆分数据集
        -   验证集需要来自input\_label.csv,每次拆时候用相同的随机种子以保证拆分结果一致
        -   训练集来自拆出来的另一部分与input\_added\_label.csv的叠加
    2.  加载模型，从模型中恢复网络权重
    3.  对训练集和验证集进行带有early stop的训练，此时需要支持没有验证集的情况
    4.  对未标注的数据进行预测
    5.  在预测结果中查询难例
    6.  保存当前阶段训练出的模型
    7.  将预测结果保存成csv文件
6.  服务端获取预测结果后让用户反馈难例的标注是否满意
    1.  如果满意，将所有预测出的标注当作数据集标注
    2.  如果不满意，将之前的手动标注数据和确认难例的标注数据合并成已标注数据
7.  如果仍有未标注数据且还未达到轮数上限，将已标注数据、剩余未标注数据、当前阶段训练的模型等传给智能标注算子继续训练，即从步骤2开始循环执行
8.  如果此次标注已结束，服务端负责将最终生成的模型提交到模型仓库，供再次进行智能标注时以此模型为起点开始训练


<a id="orgf435422"></a>

## 如何新增模型用于智能标注

1.  按照一套协议将新模型的逻辑封装到模型中，这套协议包含：
    -   如何加载模型
    -   如何保存模型
    -   如何训练
    -   如何筛选难例
2.  将模型上传到模型仓库即可在智能标注里选择使用


<a id="org3c83abb"></a>

## 生成新模型的流程


<a id="org20de915"></a>

### 准备数据文件

格式为csv，具体参照([fastlabel智能标注数据增强流程文档.md](https://gitlab.datacanvas.com/APS/public-knowledge-base/blob/master/07.Pipes%E5%BC%80%E5%8F%91%E8%A7%84%E8%8C%83/fastlabel%E6%99%BA%E8%83%BD%E6%A0%87%E6%B3%A8%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA%E6%B5%81%E7%A8%8B%E6%96%87%E6%A1%A3.md))。

每种场景均需要提供3个csv文件：

-   input\_label.csv
    
    第一次启动智能标注前的已标注样本,含有列：sampleId，path，label

-   input\_added\_label.csv
    
    启动智能标注后追加的已标注样本，含有列：sampleId，path，label

-   input\_unlabel.csv
    
    待标注样本，含有列：sampleId，path

将这三个csv文件放到 `exp_dir/input` 目录中


<a id="org176c3b4"></a>

### 在 `create_pipeline_demo.py` 中修改参数

    input_dir = "exp_dir/input"  # 里面需要有文件：input_label.csv，input_added_label.csv，input_unlabel.csv
    output_dir = "exp_dir/output"
    is_first_train = True
    query_cnt = 100
    val_size = 0.2
    strategy = "EntropySampling"
    options = {"epochs": 1,
               "batch_size": 16,
               "lr": 0.1,
               "lrf": 0.05}
    model_id = None


<a id="org925253f"></a>

### 在demo目录中开发模型

1.  在my\_model.py文件的MyModel类中实现方法：

    1.  init方法
    
        初始化模型
    
    2.  split\_train\_val
    
        入参：
        
        -   df\_init，启动智能标注时的初始标注
        -   df\_anno，启动智能标注后新增的标注
        -   is\_first\_train，是否为启动智能标注后的第一轮训练
        -   val\_size，验证集比例，为占df\_init的比例
        -   random\_seed，随机种子
        
        需要写的逻辑：
        
        1.  根据随机种子对df\_init进行训练集和验证集的拆分
        2.  如果is\_first\_train为False：将拆分出的训练集与df\_anno合并成最终训练集
        3.  训练集为df\_train, 验证集为df\_val
        4.  如果val\_size为0，代表没有验证集,此时df\_val应为None
        
        返回值：训练集和验证集
    
    3.  save\_model
    
        入参：
        
        -   save\_dir，模型要保存到的目录
        
        需要写的逻辑：将需要恢复当前模型用到的所有文件都保存到这个目录里。
    
    4.  load\_model
    
        入参：
        
        -   model\_dir，模型的加载目录
        
        需要写的逻辑：从model\_dir中加载模型，恢复到可以用于训练的状态
        
        返回值：当前类的实例
    
    5.  train\_model
    
        入参：
        
        -   df\_train，训练集，pandas的DataFrame
        -   df\_val，验证集，pandas的DataFrame，可能为None，此时代表没有验证集
        -   work\_dir，工作目录，训练过程中产生的文件都放到这个目录下
        -   is\_first\_train，是否为启动智能标注后的第一轮训练
        -   options，算子调用时传的额外参数，dict
        
        需要写的逻辑：
        
        1.  根据训练集和验证集训练训练模型
        2.  需要保证在df\_val为None的情况下能正常训练
        3.  如果为第一轮智能标注训练，可能需要根据训练数据的类别个数调整模型的结构
    
    6.  query\_hard\_example
    
        入参：
        
        -   df\_img，未标注数据，pandas的DataFrame
        -   work\_dir，工作目录，与训练时是同一个目录
        -   query\_cnt，查询难例的数目
        -   strategy，难例查询策略，默认为LeastConfidence
        -   options，算子调用时传的额外参数，dict
        
        需要写的逻辑：
        
        1.  使用train\_model中生成的模型对df\_img进行预测
        2.  实现相应难例查询策略，从预测结果中筛选难例
        
        返回值：将预测结果和难例状态一起构建成的pandas DataFrame

2.  其它文件

    可以将模型训练以及预测的逻辑封装成一个独立的package放到demo目录中，将
    my\_model.py当作一个入口去调用其它代码


<a id="org4aaccb0"></a>

### 启动create\_pipeline\_demo.py生成模型文件

