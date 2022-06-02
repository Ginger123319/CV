# replace去除的特殊字符，像是空格，但是无法用去除空格的方式去掉
import torch
import torchvision

s = "(外用)妇炎平胶囊　汕头                            "
# print(s.replace(" ", "").replace("\t", "").replace("\n", ""))
print(s.replace(chr(12288), ""))
a = ""
for i in s:
    # print(i)
    if i == " " or i == "\n" or ord(i) == 12288:
        # print("空", i)
        continue
    else:
        # print(ord(i))
        a += i
# print(chr(12288))
print(a)
print(torch.cuda.is_available())
print(torchvision.version)
