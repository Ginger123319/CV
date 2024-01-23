# -*- encoding: utf-8 -*-
import os

from pyspark.sql import SparkSession


def get_or_create_local_session(require_pmml=True):
    """创建一个本地的Session， 并覆盖单例中的Session, 如果用户设置了spark，再调用此方法会丢失用户设置的Session。

    Args:
        require_pmml:

    Returns:

    """

    spark = SparkSession._instantiatedSession
    if spark is not None:
        return spark
    else:
        spark_builder = SparkSession.builder \
            .appName("LocalSparkSessionBySparkUtil") \
            .master("local[*]")

        if require_pmml:
            JPMML_CLASSPATH = os.environ.get('JPMML_CLASSPATH')
            if JPMML_CLASSPATH is None:
                raise Exception("在单机Spark模式运行下，请设置\"JPMML_CLASSPATH\"环境变量指定jpmml依赖的地址，"
                                "见文档: https://gitlab.datacanvas.com/APS/dc-sdk-mr-py/wikis/模型仓库/模型仓库SDK安装说明,"
                                "如果不需要JPMML请设置require_pmml=False。")
            jar_path = os.path.abspath(JPMML_CLASSPATH)
            if os.path.exists(jar_path) is False:
                raise RuntimeError("配置的JPMML Jar不存在, 路径如下:\n%s" % jar_path)

            spark_builder.config("spark.executor.extraClassPath", jar_path)\
                .config("spark.driver.extraClassPath", jar_path)

        spark = spark_builder.getOrCreate()

    return spark


def get_spark_session():
    """
    支持Livy中Spark2.x版本。
    :return:
    """
    spark = SparkSession._instantiatedSession
    if spark is not None:
        return spark
    else:
        raise Exception("没有找到SparkContext, 清调用set_spark_session(session)设置，或者调用 "
                        "get_or_create_local_session()创建本地session")
        # return get_or_create_local_session()


def get_spark_context():
    return get_spark_session().sparkContext


def set_spark_session(session):
    global spark
    spark = session
