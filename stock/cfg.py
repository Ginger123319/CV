code = ["sh.600", "sh.601", "sh.603", "sh.605", "sz.000", "sz.002", "sz.300"]
for head in code:
    for i in range(1000):
        # 原有字符往右边调整，左边不足的位置补零
        print(head + str(i).rjust(3, "0"))
