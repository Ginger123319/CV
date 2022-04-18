# 训练R网络

import nets
import train

if __name__ == '__main__':
    net = nets.RNet()

    trainer = train.Trainer(net, r'D:\Python\source\FACE\celebA\save_pic_label\24\rnet.pt',
                            r"D:\Python\source\FACE\celebA\save_pic_label\24")  # 网络，保存参数，训练数据；创建训练器
    trainer.train()  # 调用训练器中的方法
