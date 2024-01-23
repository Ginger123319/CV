# -*- encoding: utf-8 -*-
from dc_model_repo.util import str_util


def convert2pmml_for_spark21(sc, input_df, pipeline_model):
    from dc_model_repo.base.mr_log import logger
    ConverterUtil = sc._jvm.org.jpmml.sparkml.ConverterUtil
    StringWriter = sc._jvm.java.io.StringWriter
    StreamResult = sc._jvm.javax.xml.transform.stream.StreamResult
    JAXBUtil = sc._jvm.org.shaded.jpmml.model.JAXBUtil
    from pyspark.ml.common import _py2java
    java_schema = _py2java(sc, input_df).schema.__call__()
    java_pipeline_model = pipeline_model._to_java()
    try:
        pmml = ConverterUtil.toPMML(java_schema, java_pipeline_model)
    except TypeError as e:
        logger.error("请检查JPMML 依赖库是否正确安装，参考：https://gitlab.datacanvas.com/APS/dc-sdk-mr-py/wikis/模型仓库/模型仓库SDK安装说明")
        raise e
    writer = StringWriter()
    JAXBUtil.marshalPMML(pmml, StreamResult(writer))
    pmml_bytes = str_util.to_bytes(writer.toString())
    writer.close()
    return pmml_bytes


def convert2pmml(sc, input_df, pipeline_model):
    from dc_model_repo.base.mr_log import logger
    spark_version = str(sc.version[:3])
    logger.info("当前Spark的版本为: %s" % spark_version)
    if spark_version in ["2.1", "2.2", "2.3", "2.4"]:
        # Spark V2.2文档: https://github.com/jpmml/pyspark2pmml/tree/spark-2.2.X, jpmml-sparkml-executable-1.3.14.jar
        # Spark V2.3文档: https://github.com/jpmml/pyspark2pmml/tree/spark-2.3.X, converter-executable-1.2.11.jar
        logger.info("当前Spark版本为: %s, 使用Spark2.2以后的方式生成pmml文件。" % spark_version)
        from dc_model_repo.pyspark2pmml import PMMLBuilder
        pmmlBuilder = PMMLBuilder(sc, input_df, pipeline_model)
        pmml_bytes = pmmlBuilder.buildByteArray()
    return pmml_bytes


# def convert2pmml(sc, input_df, pipeline_model):
#     from dc_model_repo.base.mr_log import logger
#     spark_version = str(sc.version[:3])
#     logger.info("当前Spark的版本为: %s" % spark_version)
#
#     if spark_version == "2.1":
#         # Spark2.1文档： https://github.com/jpmml/jpmml-sparkml/tree/1.2.7, converter-executable-1.2.7.jar, 支持jdk7
#         # 如果此方法报错，请确保依赖包安装正确。
#         pmml_bytes = convert2pmml_for_spark21(sc, input_df, pipeline_model)
#     else:
#         if spark_version in ["2.2", "2.3", "2.4"]:
#             # Spark V2.2文档: https://github.com/jpmml/pyspark2pmml/tree/spark-2.2.X, jpmml-sparkml-executable-1.3.14.jar
#             # Spark V2.3文档: https://github.com/jpmml/pyspark2pmml/tree/spark-2.3.X, converter-executable-1.2.11.jar
#             logger.info("当前Spark版本为: %s, 使用Spark2.2以后的方式生成pmml文件。" % spark_version)
#         from dc_model_repo.pyspark2pmml import PMMLBuilder
#         pmmlBuilder = PMMLBuilder(sc, input_df, pipeline_model)
#         pmml_bytes = pmmlBuilder.buildByteArray()
#     return pmml_bytes
