param_path = r"param_pt"
opt_path = r"opt_pt"
log_path = r"./log"
device = "cuda:0"

train_batch_size = 256
test_batch_size = 256

data_dir = r"..\..\source\stock\data"
train_dir = r"..\..\source\stock\train"
test_dir = r"..\..\source\stock\test"


code = ["sh.60", "sz.30", "sz.00"]

train_file = r"..\..\source\stock\data\train.txt"
test_file = r"..\..\source\stock\data\test.txt"

if __name__ == '__main__':
    print(1)
    for head in code:
        for i in range(1000):
            # 原有字符往右边调整，左边不足的位置补零
            print(head + str(i).rjust(4, "0"))
