import torchtext
import openpyxl
import cfg

# 加载xlsx文件
wb = openpyxl.load_workbook(cfg.file_path, read_only=True)
print(wb)
sheet = wb['Sheet1']
# print(sheet['a2'].value)
# print(type(sheet.rows))
save_file = open(cfg.save_path, "w", encoding="utf - 8")
for val in sheet.rows:
    # print(type(val))
    # print(val)
    for cel in (val[0:1] + val[2:4]):
        # print(type(cel.value))
        cel = str(cel.value).replace(" ", "").replace("\n", "").replace("\t", "").replace(chr(12288), "")
        print(cel)
        save_file.write(cel)
        save_file.write(" ")
    save_file.write("\n")
save_file.close()
print("save success!")
# one_rows = [val for val in sheet.rows][0]
# for cel in one_rows:
#     print(cel.value)
