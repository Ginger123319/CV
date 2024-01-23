# -*- encoding: utf-8 -*-
import os
import zipfile


def extract(zip_file_path, extract_dest):
    """解压缩文件

    Args:
        zip_file_path: 压缩文件地址
        extract_dest: 解压到目录，要求目录已经存在

    Returns:

    """
    if not os.path.exists(extract_dest):
        raise Exception("路径[%s]不存在。" % extract_dest)

    f = zipfile.ZipFile(zip_file_path)
    for file in f.namelist():
        f.extract(file, extract_dest)
    f.close()


def compress(dir_path, zip_path):
    """压缩文件夹。
    2021.1.20 新增对单个文件的压缩支持：入职dir_path是一个文件而非目录，会将该文件直接压缩到zip_path中
    Examples:
        dir_path结构：
         |-a.txt
         |-b.txt
        压缩生成的zip文件结构：
         |-a.txt
         |-b.txt
    Args:
        dir_path:
        zip_path:

    Returns:

    """
    zip_file = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
    if os.path.isdir(dir_path):
        for root, dirnames, filenames in os.walk(dir_path):
            file_path = root.replace(dir_path, '')  # 去掉根路径，只对目标文件夹下的文件及文件夹进行压缩
            # 循环出一个个文件名
            for filename in filenames:
                zip_file.write(os.path.join(root, filename), os.path.join(file_path, filename))
    else:
        zip_file.write(dir_path, os.path.basename(dir_path))
    zip_file.close()
