import os
from xml.dom.minidom import parse

'''
minidom.parse(filename)                     #加载和读取xml文件
doc.documentElement                         #获取xml文档对象
node.getAttribute(AttributeName)            #获取xml节点属性值
node.getElementsByTagName(TagName)          #获取xml节点对象集合
#.getElementsByTagName()，根据name查找根目录下的子节点
node.childNodes                             #获取子节点列表
node.childNodes[index].nodeValue        #获取xml节点值
node.firstChild                             #访问第一个节点
n.childNodes[0].data                        #获得文本值
node.childNodes[index].nodeValue        #获取XML节点值
doc=minidom.parse(filename)
doc.toxml('utf-8')                          #返回Node节点的xml表示的文本
'''
dirpath = r'outputs/'  # 文件地址
file_name_list = os.listdir(dirpath)  # 获取文件地址下的所有文件名称
# print(file_name)
for file_name in file_name_list:  # 循环列表中的所有文件名
    j = file_name.split('.')[0]
    xml_doc = os.path.join(dirpath, file_name)  # 拼接文件地址
    # print(xml_doc)
    dom = parse(xml_doc)  # 解析address文件，返回DOM对象，address为文件地址
    root = dom.documentElement  # 创建根节点。每次都要用DOM对象来创建任何节点,获取根节点,赋值给root作为节点名称
    img_name = root.getElementsByTagName('path')[0].childNodes[
        0].data  # 获取某个元素节点的文本内容，先获取子文本节点，然后通过“data”属性获取文本内容，这里获取图片名字
    # print(img_name)
    img_size = root.getElementsByTagName('size')[0]  # 获取图片尺寸大小，返回一个列表的迭代器
    # print(img_size)
    objects = root.getElementsByTagName('object')  # 获取项目节点名称
    img_w = img_size.getElementsByTagName('width')[0].childNodes[0].data  # 获取图片的宽
    # print(img_w)
    img_h = img_size.getElementsByTagName('height')[0].childNodes[0].data  # 获取图片的高
    img_c = img_size.getElementsByTagName('depth')[0].childNodes[0].data  # 获取图片的通道数
    # print(img_name)
    # print(img_w,img_h,img_c)
    f = open("Parse_label.txt", 'a', encoding='utf-8')  # 打开txt文件
    f.writelines("images/%s.jpg" % j)  # 将"images/%s.jpg" % j写入txt文件
    for box in objects:  # 循环objects的所有标注的框
        # print(len(box.getElementsByTagName("name")))
        # print(len(box.getElementsByTagName("name")))
        for i in range(len(box.getElementsByTagName("name"))):  # 循环框的类别名字的长度方便下面判断
            cls_name = box.getElementsByTagName('name')[i].childNodes[0].data  # 获取框的类别的名字
            if cls_name == '人':  # 判断名字和分类是否相同
                cls_num = 0  # 给每类编号
            elif cls_name == '猫':
                cls_num = 1
            elif cls_name == '狗':
                cls_num = 2
            else:
                cls_num = 3
            # print(cls_name,cls_num)
            x1 = int(box.getElementsByTagName('xmin')[i].childNodes[0].data)  # 获取框的左上角x1的坐标
            y1 = int(box.getElementsByTagName('ymin')[i].childNodes[0].data)  # 获取框的左上角y1的坐标
            x2 = int(box.getElementsByTagName('xmax')[i].childNodes[0].data)  # 获取框的右下角角x2的坐标
            y2 = int(box.getElementsByTagName('ymax')[i].childNodes[0].data)  # 获取框的右下角角y2的坐标
            print(cls_name, (x1, y1, x2, y2))
            box_w = x2 - x1  # 计算框的宽高
            box_h = y2 - y1
            cx = int(x1 + box_w / 2)  # 计算框的中心点坐标并转成int类型
            cy = int(y1 + box_h / 2)
            print(cls_name, (cx, cy, box_w, box_h))
            f.writelines(" {} {} {} {} {} \t".format(cls_num, cx, cy, box_w, box_h))  # 按行将起写入txt文件
    f.writelines("\n")  # 换行
    f.flush()  #
    # flush() 方法是用来刷新缓冲区的，即将缓冲区中的数据立刻写入文件，同时清空缓冲区，不需要是被动的等待输出缓冲区写入。
    # 一般情况下，文件关闭后会自动刷新缓冲区，但有时你需要在关闭前刷新它，这时就可以使用 flush() 方法。
    f.close()
