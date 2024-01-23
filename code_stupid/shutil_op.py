import shutil
import os

# 加入dirs这个参数，虽然不会报错，但是也不会清空目录；使得copy后的数据含有历史残留
shutil.copytree(r'D:\Python\test_jzyj\augment', r'D:\Python\test_jzyj\augmen', dirs_exist_ok=True)
os.makedirs(r'D:\Python\test_jzyj\augmen', exist_ok=True)
