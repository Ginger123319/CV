dst = open(r"G:\liewei\source\word_vec\atc测试 - 副本.txt", 'w', encoding="utf-8")
with open(r"G:\liewei\source\word_vec\atc测试.txt", 'r', encoding="utf-8") as f:
    f.readline()
    for line in f.readlines():
        print(line)
        dst.write(line.replace(" ", "|"))
dst.close()
