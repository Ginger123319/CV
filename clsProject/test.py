import os
import shutil

import torch
from torch.nn import BCELoss

# from dataset import MyData
from dataset_changed import MyData
# from net import Net
from net2 import Net
from torch.utils.data import DataLoader

param_path = r"param_pt"
mask_path = r"..\..\source\enzyme\mask_result\mask.txt"
s_sequence = "0GAVLIPFYWSTCMNQDEKRHX"

test_loader = DataLoader(MyData(False), batch_size=48, shuffle=False)
net = Net()
loss_fun = BCELoss()
# 加载参数
if os.path.exists(param_path):
    try:
        net.load_state_dict(torch.load(param_path))
        print("Loaded!")
    except RuntimeError:
        os.remove(param_path)
        print("参数异常，重新开始训练")
else:
    print("No Param!")
sum_test_loss = 0.
sum_score = 0.
for epoch in range(1):
    for i, (test_data, test_tag, real_data) in enumerate(test_loader):
        net.eval()
        # 新加的
        test_data = test_data.permute(0, 3, 1, 2)
        out = net(test_data)[0]
        out1 = net(test_data)[1]
        out2 = net(test_data)[2]
        loss = loss_fun(out.reshape(-1), test_tag.float())

        # 精度计算
        score = torch.mean((torch.eq((out.reshape(-1) > 0.5).float(), test_tag.float())).float())
        print("out:{}\ntag:{}".format((out.reshape(-1)).float(), test_tag.float()))

        # print(torch.mean((torch.eq((out.squeeze() > 0.5).float(), test_tag.float())).float()))
        # print(test_tag.float())
        # exit()
    # print("epoch {} score {} ".format(epoch, score))
    sum_test_loss += loss.item()
    sum_score += score.item()
    # print(out1.permute(0, 2, 3, 1).shape)
    # print(out1.permute(0, 2, 3, 1))
    # print(torch.argmax(out1.permute(0, 2, 3, 1)).shape)
    mask_out = out1
    print(mask_out.shape)
    for i in range(mask_out.shape[0]):
        print(mask_out[i].squeeze().shape)
        _mask_out = mask_out[i].squeeze()
        # print(_mask_out[0].shape)
        sum_list = []
        for j in range(_mask_out.shape[0]):
            idx = torch.argmax(_mask_out[j])
            print(idx)
            result = real_data[i][idx - 2:idx + 3]
            print(result)
            sum_list.append(result+" ")
        # for _, elem in enumerate(mask_out[i]):
        #     print(elem.squeeze().shape)
        #     print(torch.argmax(elem.squeeze()))
        #     idx = torch.argmax(elem.squeeze())
        #     # result = test_data.permute(0, 2, 3, 1)[i][0][idx - 2:idx + 3]
        # result1 = real_data[i][idx1 - 2:idx1 + 3]
        # result2 = real_data[i][idx2 - 2:idx2 + 3]
        # print(result1 + " " + result2)
        # result = result1 + " " + result2
        # test_data = test_data.permute(0, 2, 3, 1)
        # print(test_data.shape)
        # mask_data = mask_out * test_data
        # # print(mask_data.shape)
        # for i in range(mask_data.shape[0]):
        #     # print(i)
        #     # break
        #     mask_list = []
        #     for index, elem in enumerate(mask_data[i][0]):
        #         # print(len(torch.nonzero(elem)))
        #         length = len(torch.nonzero(elem))
        #         if length > 0:
        #             s_index = torch.nonzero(elem).item()
        #             # mask_list.append(str(index))
        #             mask_list.append(s_sequence[s_index])
        #         else:
        #             # mask_list.append(str(index))
        #             mask_list.append("_")
        print(sum_list)
        with open(mask_path, 'a') as m:
            m.write("".join(sum_list))
            m.write("\n")

# 测试
test_avg_loss = sum_test_loss / (epoch + 1)
test_avg_score = sum_score / (epoch + 1)
print("{} epochs: test_avg_loss is {}".format((epoch + 1), test_avg_loss))
print("{} epochs: test_avg_score is {}".format((epoch + 1), test_avg_score))
