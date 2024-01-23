# -*- coding: utf-8 -*-
import uuid, os

# __fs__ = None


def __get_fs__(sc):
    # global __fs__
    # if __fs__ is None:
    FileSystem = sc._gateway.jvm.org.apache.hadoop.fs.FileSystem
    __fs__ = FileSystem.get(sc._jsc.hadoopConfiguration())
    return __fs__


def __make_path__(sc, p):
    Path = sc._gateway.jvm.org.apache.hadoop.fs.Path
    return Path(p)


def write_file_bytes(sc, path, content):
    file_name = '/tmp/tmp-%s.file' % str(uuid.uuid4())
    # print('写入到临时文件: %s' % file_name)
    # 1. write file to local
    with open(file_name, mode='wb') as f:
        f.write(content)
    src_file = __make_path__(sc, file_name)
    dest = __make_path__(sc, path)
    fs = __get_fs__(sc)
    fs.copyFromLocalFile(src_file, dest)
    fs.close()


def read_bytes(sc, path):
    """
    将hdfs中的数据读取成字节放到内存中， 只适合读取小文件。
    :param sc:
    :param path:
    :return:
    """
    # 1. 把文件下载到本地
    file_tmp = '/tmp/tmp-%s.file' % str(uuid.uuid4())
    dest = __make_path__(sc, file_tmp)
    src = __make_path__(sc, path)
    fs = __get_fs__(sc)
    # public void copyToLocalFile(boolean delSrc,
    #                    Path src,
    #                    Path dst,
    #                    boolean useRawLocalFileSystem)
    #                      throws IOException
    # The src file is under FS, and the dst is on the local disk.
    # Copy it from FS control to the local dst name. delSrc indicates if the src will be removed or not.
    # useRawLocalFileSystem indicates whether to use RawLocalFileSystem as local file system or not.
    # RawLocalFileSystem is non crc file system.So, It will not create any crc files at local.
    # Parameters:
    # delSrc - whether to delete the src
    # src - path
    # dst - path
    # useRawLocalFileSystem - whether to use RawLocalFileSystem as local file system or not.
    fs.copyToLocalFile(False, src, dest, True)
    fs.close()

    # 2. 从本地读取字节
    buf = bytearray(os.path.getsize(file_tmp))
    with open(file_tmp, 'rb') as f:
        f.readinto(buf)

    return bytes(buf)


def copy_from_local_file(sc, src, dest):
    p_src = __make_path__(sc, src)
    p_dest = __make_path__(sc, dest)
    fs = __get_fs__(sc)
    fs.copyFromLocalFile(p_src, p_dest)
    fs.close()


def copy_to_local_file(sc, src, dest):
    p_src = __make_path__(sc, src)
    p_dest = __make_path__(sc, dest)
    fs = __get_fs__(sc)
    fs.copyToLocalFile(False, p_src, p_dest, True)
    fs.close()


def exists(sc, path):
    fs = __get_fs__(sc)
    ret = fs.exists(__make_path__(sc, path))
    fs.close()
    return ret


def list_files(sc, path):
    fs = __get_fs__(sc)
    p_hdfs = __make_path__(sc, path)
    file_status_list = fs.listStatus(p_hdfs)
    file_list = []
    for file_status in file_status_list:
        if file_status.isDirectory():
            f_name = file_status.getPath().getName()
            file_list.append(f_name)
    fs.close()
    return file_list


def copy(sc, src, dest):
    FileUtil = sc._gateway.jvm.org.apache.hadoop.fs.FileUtil
    fs = __get_fs__(sc)
    p_src = __make_path__(sc, src)
    p_dest = __make_path__(sc, dest)
    configuration = sc._jsc.hadoopConfiguration()
    FileUtil.copy(fs, p_src, fs, p_dest, False, configuration)
    fs.close()


def move(sc, src, dest):
    fs = __get_fs__(sc)
    p_src = __make_path__(sc, src)
    p_dest = __make_path__(sc, dest)
    fs.rename(p_src, p_dest)
    fs.close()


def delete(sc, path):
    fs = __get_fs__(sc)
    # 递归删除
    fs.delete(__make_path__(sc, path), True)
    fs.close()


def make_dirs(sc, path):
    fs = __get_fs__(sc)
    fs.mkdirs(__make_path__(sc, path))
    fs.close()


def is_dir(sc, path):
    fs = __get_fs__(sc)
    ret = fs.isDirectory(__make_path__(sc, path))
    fs.close()
    return ret


def is_file(sc, path):
    fs = __get_fs__(sc)
    ret = fs.isFile(__make_path__(sc, path))
    fs.close()
    return ret
