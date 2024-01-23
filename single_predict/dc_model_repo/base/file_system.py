# -*- encoding: utf-8 -*-

import os
from os import path as P

import abc
import shutil
import six

from dc_model_repo.util import hdfs_client
from dc_model_repo.util import spark_util


class DCFileSystem(object):

    @abc.abstractmethod
    def get_name(self):
        raise NotImplemented

    @abc.abstractmethod
    def delete_dir(self, p):
        pass

    @abc.abstractmethod
    def delete_file(self, p):
        pass

    @abc.abstractmethod
    def copy_from_local(self, src, dest):
        pass

    @abc.abstractmethod
    def copy(self, src, dest):
        """复制文件，将文件或者目录拷贝到指定目录。当dest不存在时候将会在复制后改名。

        ----
        假设存在有文件：
          - `/tmp/model/model.pkl`
          - `/tmp/step`
        Case1: src是文件， dest是存在的目录
          当源文件为:`/tmp/model/model.pkl`，目的目录为：`/tmp/step`，复制结果为：`/tmp/step/mode.pkl`

        Case2: src是文件，dest不存在
          当源文件为:`/tmp/model/model.pkl`，目的路径为：`/tmp/step/my_model.pkl`，复制结果为：`/tmp/step/my_model.pkl`

        Case3: src是目录，dest存在
          当源文件为:`/tmp/model`，目的目录为：`/tmp/step`，复制结果为：`/tmp/step/model/model.pkl`

        Case4: src是目录，dest不存在
          当源文件为:`/tmp/model`，目的目录为：`/tmp/step/my_step`，复制结果为：`/tmp/step/my_step/model.pkl`

        Args:
            src(str): 源路径，可以是文件或者地址。
            dest(str): 目的目录，如果不存在将会新建。

        """
        pass

    @abc.abstractmethod
    def make_dirs(self, p):
        pass

    @abc.abstractmethod
    def write_bytes(self, p, data):
        pass

    @abc.abstractmethod
    def exists(self, p):
        pass

    @abc.abstractmethod
    def read_bytes(self, p):
        pass

    @abc.abstractmethod
    def listdir(self, p):
        pass

    @abc.abstractmethod
    def rename(self, before, after):
        pass

    @abc.abstractmethod
    def is_dir(self, p):
        pass

    @abc.abstractmethod
    def is_file(self, p):
        pass


class HDFSFileSystem(DCFileSystem):

    def is_file(self, p):
        return hdfs_client.is_file(spark_util.get_spark_context(), p)

    def is_dir(self, p):
        return hdfs_client.is_dir(spark_util.get_spark_context(), p)

    def rename(self, before, after):
        hdfs_client.move(spark_util.get_spark_context(), before, after)

    def copy(self, src, dest):
        return hdfs_client.copy(spark_util.get_spark_context(), src, dest)

    def listdir(self, p):
        return hdfs_client.list_files(spark_util.get_spark_context(), p)

    def get_name(self):
        return 'hdfs'

    def read_bytes(self, p):
        return hdfs_client.read_bytes(spark_util.get_spark_context(), p)

    def exists(self, p):
        return hdfs_client.exists(spark_util.get_spark_context(), p)

    def delete_dir(self, p):
        return hdfs_client.delete(spark_util.get_spark_context(), p)

    def delete_file(self, p):
        return hdfs_client.delete(spark_util.get_spark_context(), p)

    def copy_from_local(self, src, dest):
        return hdfs_client.copy_from_local_file(spark_util.get_spark_context(), src, dest)

    def make_dirs(self, p):
        return hdfs_client.make_dirs(spark_util.get_spark_context(), p)

    def write_bytes(self, p, data):
        return hdfs_client.write_file_bytes(spark_util.get_spark_context(), p, data)


class LocalFileSystem(DCFileSystem):

    def is_dir(self, p):
        return os.path.isdir(p)

    def is_file(self, p):
        return os.path.isfile(p)

    def rename(self, before, after):
        # 最好是同级别目录
        os.rename(before, after)

    def copy(self, src, dest):
        if os.path.isdir(src):
            # copytree 要求dest是不存在，如果存在就在目的路径下新建一个目录
            if P.exists(dest):
                dest = P.join(dest, P.basename(src))
            shutil.copytree(src, dest)
        else:
            # 复制一个会现在新目录创建文件名称
            shutil.copy(src, dest)

    def listdir(self, p):
        return os.listdir(p)

    def get_name(self):
        return 'local'

    def read_bytes(self, p):
        buf = bytearray(os.path.getsize(p))
        with open(p, 'rb') as f:
            f.readinto(buf)
        return bytes(buf)

    def exists(self, p):
        return os.path.exists(p)

    def delete_dir(self, p):
        import shutil
        shutil.rmtree(p, ignore_errors=True)

    def delete_file(self, p):
        os.remove(p)

    def copy_from_local(self, src, dest):
        shutil.copy(src, dest)

    def make_dirs(self, p):
        if not os.path.exists(p):
            if six.PY3:
                os.makedirs(p, exist_ok=True)
            else:
                os.makedirs(p)

    def write_bytes(self, p, data):
        with open(p, 'wb') as f:
            f.write(data)


def instance_by_name(kind):
    if kind == 'hdfs':
        # from dc_model_repo.step.file_system import HDFSFileSystem
        return new_hdfs_fs()
    elif kind == 'local':
        # from dc_model_repo.step.file_system import LocalFileSystem
        return new_local_fs()
    else:
        raise Exception("不支持类型是%s的文件系统%s" % kind)


def new_hdfs_fs():
    return HDFSFileSystem()


def new_local_fs():
    return LocalFileSystem()


FS_LOCAL = 'local'
FS_HDFS = 'hdfs'
