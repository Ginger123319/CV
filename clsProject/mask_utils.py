mask_path = r"..\..\source\enzyme\mask_result\mask.txt"
save_path = r"..\..\source\enzyme\mask_result\save.txt"
str_list = []
with open(mask_path, 'r') as m:
    for line in m:
        # print(line.split("_"))
        for c in line.split("_"):
            if len(c) > 1:
                # print(c)
                str_list.append("_" + c)
        str_list.append("\n")
with open(save_path, 'a') as s:
    # print(str_list)
    s.write("".join(str_list))
