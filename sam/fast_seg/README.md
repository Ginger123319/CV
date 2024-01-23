# 更新内容
- 更新模型为SAM-HQ
- 修改app.py等代码逻辑
- 新增支持box作为prompt提示辅助掩码生成
- 需要修改的地方：新增参数处理缓存可用内存大小以及分割模式参数，默认为SAM
- 前端需要新增通过box触发请求等
- 支持将图片经过image encoder输出的特征缓存到内存中
- 后续支持：text作为prompt等


# 项目介绍
本项目图像分割辅助标注功能。

- 输入：原始图像、初始mask、提示点击（正例点击和负例点击）
- 输出：预测的mask
- 默认使用SAM模型进行分割，如果需要精细分割，需要选择SAM-HQ模式

# 环境依赖
在base-image-dl-gpu基础镜像上安装requirements.txt中的包。

或使用最小依赖包：
```
tornado==6.1
opencv-python==4.6.0.66
scipy
numpy
Cython
albumentations==0.5.2
tqdm
pyyaml
easydict==1.9
torch==1.10.2
torchvision==0.11.3
cachetools
timm
```

# 启动服务
python app.py

# 接口文档
## 基本信息
 - 请求类型：POST
 - URL: http://localhost:8888/seg/magic 
 - Content-Type：application/json

##  Body参数及说明
```
{
    "imgPath": "/mnt/aps/..../uuid/image1.png",
    "preMask": "/mnt/aps/..../uuid/mask1.png",
    "resultPath": "/mnt/aps/.../result1.png",
    "box": {
        "xmin": 200,
        "ymin": 50,
        "xmax": 600,
        "ymax": 200
    },
    "maskColor": {
        "R": 0,
        "G": 255,
        "B": 0,
        "A": 200
    },
    "click": [
        {
            "x": 410.1,
            "y": 116.2,
            "is_positive": true
        },
        {
            "x": 320.9015151515151,
            "y": 90.23484848484847,
            "is_positive": true
        },
        {
            "x": 368.9015151515151,
            "y": 102.23484848484847,
            "is_positive": false
        }
    ]
}
```
注意：
1. 第一次调用时没有preMask,此时preMask节点填空字符串或null
2. 返回mask文件名字为：mask_{毫秒时间戳}_{长度为5的随机字符串}.png
3. click的list长度必须大于等于1，list中的点需要按照点击先后顺序排列，第一个点必须是positive为true的点
4. 如果想让本服务自动推荐maskColor，在传参数时候maskColor的"R","G","B"对应的值均填-1，"A"还正常传，此时本服务会根据点击及识别情况自动设置mask的颜色。当"R","G","B"不是-1时，会按照传入的颜色返回mask。所以推荐在调用本接口时这么做：

      1. 识别当前物体的第一次点击时"R","G","B"对应的值均填-1，"A"还正常传，此时服务返回推荐的颜色 
      2. 识别当前物体的后续点击时"R","G","B"使用上次返回的mask中的颜色，这样能保证同一个物体的mask颜色一致

## 响应示例
1. 执行成功
    ```
    {
        "code": 0,
        "data": {
            "mask": "/home/zk/code/fast_seg/image_1661493142.png"，
            "maskColor": {
                "R": 0,
                "G": 255,
                "B": 0,
                "A": 200
            },
            "bndbox": {
                "xmin": 5,
                "ymin": 6,
                "xmax": 100,
                "ymax": 106
            }
        },
        "msg": "ok"
    }
    ```
    其中maskColor返回mask中的RGBA值
2. 内部异常
    ```
    {
        "code": -1,
        "msg": "ERROR message."
    }
    ```


