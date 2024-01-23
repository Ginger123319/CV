# -*- encoding: utf-8 -*-
import time

import requests
import os
from os import path as P

from dc_model_repo.util import str_util, json_util, validate_util, zip_util

# 链接超时时间，读取超时时间
TIMEOUT = (30, 60)  #

dc_proxy = None


def get_dc_proxy():
    """在工作流执行过程中获取运行环境

    Returns: 运行时代理，可从中获取工作流环境配置

    """
    from workflow_collector.workflow_collector import RuntimeProxy
    global dc_proxy
    if dc_proxy is None:
        dc_proxy = RuntimeProxy()
    return dc_proxy


def _get_mr_server_portal():
    from dc_model_repo.base import collector_manager
    extension_collector = collector_manager.get_extension_collector()
    if extension_collector is not None:
        return extension_collector.get_model_repo_name(), extension_collector.get_mr_server_portal()
    else:
        return None, None


def submit_and_publish(model_path, model_group_type_name, model_group_name, is_DCPipeline_model=True,
                       model_format=None, framework=None, model_name=None, description=None,
                       algorithm_name=None, timeout=TIMEOUT):
    """
    将本地DCPipeline模型上传到模型仓库并发布到模型组

    Args:
        model_path: 本地模型路径
        model_group_type_name: 模型组类型。目前有：机器学习模型、深度学习模型、预训练模型
        model_group_name: 模型组名称。该模型组必须为已存在的模型组。
        is_DCPipeline_model: 是否是DCPipeline模型。 现在不支持非DCPipeline，只能填True.
        model_format: 模型格式（只有非DCPipeline模型时需要填写）。可填：zip、pkl、pmml、onnx、h5、pb、ckpt、saved_model、pth
        framework: 模型框架（只有非DCPipeline模型时需要填写）。可填：DeepTables、Keras、TensorFlow、TensorFlow2、Pytorch、SKLearn
        model_name: 模型名称（只有非DCPipeline模型时需要填写）。不能与目标模型组中的重复。
        description: 模型描述（只有非DCPipeline模型时需要填写）。
        algorithm_name: 算法名称（只有非DCPipeline模型时需要填写）。
        timeout: 超时时间
    """
    assert is_DCPipeline_model, "不支持非DCPipeline模型的发布。"
    from dc_model_repo.base.mr_log import logger
    # 上传模型
    if is_DCPipeline_model:
        model_id = submit(model_path, timeout=timeout)
    else:
        lacked_params = []
        if model_format is None:
            lacked_params.append("model_format")
        if framework is None:
            lacked_params.append("framework")
        if model_name is None:
            lacked_params.append("model_name")
        if description is None:
            lacked_params.append("description")
        if algorithm_name is None:
            lacked_params.append("algorithm_name")
        if len(lacked_params) > 0:
            raise Exception("当上传非DCPipeline模型时，这些参数必须填写：{}".format(", ".join(lacked_params)))
        model_id = submit_custom_model(model_path, model_format, framework, model_name, description, algorithm_name, timeout=timeout)
    # 发布模型
    publish_model(model_ids=[model_id], model_group_type_name=model_group_type_name, model_group_name=model_group_name, timeout=timeout)
    logger.info("位于[{}]的模型已上传并发布。".format(model_path))


def publish_model(model_ids, model_group_type_name, model_group_name, timeout=TIMEOUT):
    """
    将Training模型仓库中的模型发布到模型组

    Args:
        model_ids: 模型ID列表
        model_group_type_name: 模型组类型
        model_group_name: 模型组名称
        timeout: 超时时间

    Returns: None

    """
    dc_proxy = get_dc_proxy()

    from dc_model_repo.base.mr_log import logger
    start_time = time.time()
    logger.info("开始调用发布到模型组接口...")
    from dc_model_repo.base import collector_manager
    extension_collector = collector_manager.get_extension_collector()
    server_portal = extension_collector.get_mr_server_portal()
    assert server_portal is not None, "无法获取接口地址！"

    publish_url = "{}/aps/internal/workflow/release/modelgroup/".format(server_portal)
    logger.info("发布到模型组接口地址：{}".format(publish_url))
    headers = {'tenantId': dc_proxy.get_tenant_id(), 'projectId': dc_proxy.get_project_id(), 'Content-type': 'application/json'}
    data = {
        "modelGroupName": model_group_name,
        "modelGroupTypeName": model_group_type_name,
        "modelIds": model_ids,
        'tenantId': dc_proxy.get_tenant_id(),
        'projectId': dc_proxy.get_project_id()}
    body = json_util.to_json_str(data).encode('utf-8')
    logger.info("要发布的模型信息：{}".format(str(body)))
    logger.info("请求头：{}".format(str(headers)))

    publish_response = requests.post(publish_url, data=body, timeout=timeout, headers=headers)

    logger.info("调用结束。耗时：{:.3f}s".format(time.time() - start_time))

    if publish_response.status_code == 200:
        text_response = str_util.to_str(publish_response.text)
        logger.info("调用发布到模型组接口的返回结果: {}".format(text_response))
        response_dict = json_util.to_object(text_response)
        if response_dict["code"] == 0:
            success = response_dict["data"]["success"]
            if len(success) > 0:
                logger.info("模型{}发布成功!".format([x["name"] for x in success]))
            failed = response_dict["data"]["failed"]
            if len(failed) > 0:
                for x in failed:
                    logger.error("模型[{}]发布失败：{}".format(x["name"], x["msg"]))
                raise Exception("模型发布失败：{}".format(repr(response_dict)))
        else:
            raise Exception("pipes的发布到模型组接口内部异常：{}".format(repr(response_dict)))
    else:
        publish_response.raise_for_status()
        raise Exception("调用pipes的发布到模型组接口失败！{}".format(repr(publish_response)))


def submit(pipeline_path, repo_name=None, model_repo_server_portal=None, timeout=TIMEOUT, **kwargs):
    """将本地模型提交DCPipeline模型到模型仓库。

    Args:
        pipeline_path: DCPipeline序列化后的文件夹。
        repo_name: 仓库名称，在APS工作流中可为空，会自动获取。
        model_repo_server_portal: 模型仓库服务的访问地址，在APS工作流中可为空，会自动获取。
        timeout: 可选timeout，超时时间。
    Returns: 模型ID
    """
    from dc_model_repo.base.mr_log import logger
    default_repo_name, default_model_repo_server_portal = _get_mr_server_portal()

    # 1. 参数读取校验
    if model_repo_server_portal is None:
        model_repo_server_portal = default_model_repo_server_portal
        if model_repo_server_portal is None:
            raise Exception("参数model_repo_server_portal不能为空。")

    if repo_name is None:
        repo_name = default_repo_name
        if repo_name is None:
            raise Exception("参数repo_name不能为空。")
    logger.info("方法参数使用repo_name=%s, model_repo_server_portal=%s" % (repo_name, model_repo_server_portal))

    if not P.exists(pipeline_path):
        raise Exception("模型目录不存在：%s" % pipeline_path)

    if not os.path.isdir(pipeline_path):
        raise Exception("参数pipeline_path=%s需要是DCPipeline序列化后的文件夹，当前传入的路径不是文件夹。" % pipeline_path)

    remote_model_path = _upload_temp_file(pipeline_path, model_repo_server_portal, timeout=timeout)

    # 3.1. 模型入库
    t_submit_start = time.time()
    model_id = _submit2server_pipeline_model(remote_model_path, repo_name, model_repo_server_portal, timeout=timeout)
    t_submit_end = time.time()
    logger.info("提交到仓库成功模型id=%s，耗时: %s（毫秒）" % (model_id, round((t_submit_end - t_submit_start) * 1000, 2)))
    return model_id


def submit_custom_model(model_path, model_format, framework, model_name, description, algorithm_name,
                        repo_name=None, model_repo_server_portal=None, timeout=TIMEOUT):
    """
    将本地非DCPipeline模型提交到模型仓库。

    Args:
        model_path:  非DCPipeline模型的模型路径，可以是文件或者目录。
        model_format: 模型格式
        framework: 模型框架
        model_name: 模型名称
        description: 模型描述
        algorithm_name: 算法名称
        repo_name: 仓库名称，在APS工作流中可为空，会自动获取。
        model_repo_server_portal: 模型仓库服务的访问地址，在APS工作流中可为空，会自动获取。
        timeout: 可选timeout，超时时间。

    Returns: 模型ID
    """
    from dc_model_repo.base.mr_log import logger
    default_repo_name, default_model_repo_server_portal = _get_mr_server_portal()

    # 1. 参数读取校验
    if model_repo_server_portal is None:
        model_repo_server_portal = default_model_repo_server_portal
        if model_repo_server_portal is None:
            raise Exception("参数model_repo_server_portal不能为空。")

    if repo_name is None:
        repo_name = default_repo_name
        if repo_name is None:
            raise Exception("参数repo_name不能为空。")
    logger.info("方法参数使用repo_name=%s, model_repo_server_portal=%s" % (repo_name, model_repo_server_portal))

    if not P.exists(model_path):
        raise Exception("模型目录不存在：%s" % model_path)

    remote_model_path = _upload_temp_file(model_path, model_repo_server_portal, timeout=timeout)

    # 3.1. 模型入库
    t_submit_start = time.time()
    filename = os.path.basename(model_path)
    framework = framework

    model_id = _submit2server_custom_model(remote_model_path=remote_model_path,
                                           repo_name=repo_name,
                                           model_repo_server_portal=model_repo_server_portal,
                                           timeout=TIMEOUT,
                                           model_format=model_format,
                                           filename=filename,
                                           framework=framework,
                                           model_name=model_name,
                                           description=description,
                                           algorithm_name=algorithm_name)
    logger.info("model_id: {}".format(model_id))

    t_submit_end = time.time()
    logger.info("提交到仓库成功模型id=%s，耗时: %s（毫秒）" % (model_id, round((t_submit_end - t_submit_start) * 1000, 2)))
    return model_id


def import_model(model_path, model_group_type_name, model_group_name, is_DCPipeline_model=True,
                 model_format=None, framework=None, model_name=None, description=None,
                 algorithm_name=None, algorithm_type=None, timeout=TIMEOUT):
    """
    将本地模型导入到模型组

    Args:
        model_path: 本地模型路径
        model_group_type_name: 模型组类型。目前有：机器学习模型、深度学习模型、预训练模型。其中DCPipeline模型只能选“机器学习模型”或“深度学习模型”，非DCPipeline模型只能选“预训练模型”
        model_group_name: 模型组名称。该模型组必须为已存在的模型组。
        is_DCPipeline_model: True or False. 是否是DCPipeline模型。
        model_format: 模型格式（只有非DCPipeline模型时需要填写，且此时必填）。可填：zip、pkl、pmml、onnx、h5、pb、ckpt、saved_model、pth
        framework: 模型框架（只有非DCPipeline模型时需要填写，且此时必填）。可填：DeepTables、Keras、TensorFlow、TensorFlow2、Pytorch、SKLearn
        model_name: 模型名称（只有非DCPipeline模型时需要填写，且此时必填）。不能与目标模型组中的重复。
        description: 模型描述（只有非DCPipeline模型时需要填写）。
        algorithm_name: 算法名称（只有非DCPipeline模型时需要填写）。
        algorithm_type: 算法名称（只有非DCPipeline模型时需要填写）。 可填：MULTICLASSIFY、BINARYCLASSIFY、REGRESSION、CLUSTERING、UNKNOWN
        timeout: 超时时间

    """
    from dc_model_repo.base.mr_log import logger
    # 校验参数
    if not P.exists(model_path):
        raise Exception("模型目录不存在：%s" % model_path)

    # DCPipeline 模型直接调发布接口
    if is_DCPipeline_model:
        assert model_group_type_name in ["机器学习模型", "深度学习模型"], "DCPipeline模型model_group_type_name只能选“机器学习模型”或“深度学习模型”"
        submit_and_publish(model_path, model_group_type_name, model_group_name, is_DCPipeline_model=True)
        return
    assert model_group_type_name in ["预训练模型"], "非DCPipeline模型model_group_type_name只能选“预训练模型”"
    # 非DCPipeline 先上传到中间位置再调用导入接口
    lacked_params = []
    if model_format is None:
        lacked_params.append("model_format")
    if framework is None:
        lacked_params.append("framework")
    if model_name is None:
        lacked_params.append("model_name")
    if description is None:
        description = ""
    if algorithm_name is None:
        algorithm_name = ""
    if algorithm_type is None:
        algorithm_type = "UNKNOWN"
    if len(lacked_params) > 0:
        raise Exception("当上传非DCPipeline模型时，这些参数必须填写：{}".format(", ".join(lacked_params)))

    # 获取接口地址前缀
    from dc_model_repo.base import collector_manager
    extension_collector = collector_manager.get_extension_collector()
    server_portal = extension_collector.get_mr_server_portal()
    assert server_portal is not None, "无法获取接口地址前缀！"
    logger.info("获取server_portal=%s" % server_portal)

    # 上传到中间位置
    remote_model_path = _upload_temp_file(model_path, server_portal, timeout=timeout)

    # 调用导入接口
    logger.info("开始准备调用导入接口所需参数...")
    dc_proxy = get_dc_proxy()
    import_url = "{}/aps/internal/model/group/import/nodcpipeline".format(server_portal)
    logger.info("导入到模型组接口地址：{}".format(import_url))
    headers = {'tenantId': dc_proxy.get_tenant_id(), 'projectId': dc_proxy.get_project_id(), 'Content-type': 'application/json'}
    data = {
        "modelGroupName": model_group_name,
        "modelGroupTypeName": model_group_type_name,
        'tenantId': dc_proxy.get_tenant_id(),
        'projectId': dc_proxy.get_project_id(),
        "path": remote_model_path,
        "fileName": os.path.basename(model_path),
        "framework": framework,
        "fileFormat": model_format,
        "name": model_name,
        "description": description,
        "algorithmType": algorithm_type,
        "frameworkVersion": "0",
        "modelLangType": 0,
        "algorithmName": algorithm_name,
        "modelType": "TensorFlow2"
    }
    body = json_util.to_json_str(data).encode('utf-8')
    logger.info("要导入的模型信息：{}".format(str(body)))
    logger.info("请求头：{}".format(str(headers)))

    start_time = time.time()
    logger.info("开始调用导入到模型组接口...")
    import_response = requests.post(import_url, data=body, timeout=timeout, headers=headers)
    logger.info("调用结束。耗时：{:.3f}s".format(time.time() - start_time))

    if import_response.status_code == 200:
        text_response = str_util.to_str(import_response.text)
        logger.info("调用导入到模型组接口的返回结果: {}".format(text_response))
        response_dict = json_util.to_object(text_response)
        if response_dict["code"] == 0:
            logger.info("模型导入成功：{}".format(str(response_dict)))
        else:
            raise Exception("pipes的导入到模型组接口内部异常：{}".format(repr(response_dict)))
    else:
        import_response.raise_for_status()
        raise Exception("调用pipes的导入到模型组接口失败！{}".format(repr(import_response)))


def _upload_temp_file(file_path, model_repo_server_portal, timeout):
    """
    将file_path目录下的文件打包上传到model_repo_server_portal对应的服务器

    Args:
        file_path: 要上传的文件或目录
        model_repo_server_portal: 远程地址
        timeout: 超时时间

    Returns: 上传请求返回值

    """
    dc_proxy = get_dc_proxy()

    from dc_model_repo.base.mr_log import logger
    logger.info("开始上传本地模型文件...")
    # 本地文件压缩成zip
    archive_path = '/tmp/%s.zip' % os.path.basename(file_path)
    zip_util.compress(file_path, archive_path)

    if os.path.exists(archive_path):
        logger.info("压缩文件成功，地址:%s" % archive_path)
    else:
        raise Exception("没有在%s生成压缩文件。" % archive_path)

    # 上传模型zip到临时目录并入库
    files = {
        "field1": (os.path.basename(archive_path), open(archive_path, 'rb'))
    }
    upload_url = "%s/aps/upload/repo" % model_repo_server_portal
    logger.info("上传文件URL: %s" % upload_url)
    t_upload_start = time.time()
    # 优先使用requests_toolbelt 的api，来支持大文件上传
    multipart_data = None
    try:
        from requests_toolbelt import MultipartEncoder
        multipart_data = MultipartEncoder(fields=files, boundary='---------------------------7de1ae242c06ca')
    except Exception as e:
        logger.warning("尝试使用requests_toolbelt库失败，可能没有安装，将使用普通模式上传，大文件有可能失败。", e)

    headers = {'tenantId': dc_proxy.get_tenant_id(), 'projectId': dc_proxy.get_project_id(), 'userId': dc_proxy.get_user_id()}
    if multipart_data is not None:
        headers['Content-Type'] = multipart_data.content_type
        upload_response = requests.post(upload_url, data=multipart_data, timeout=timeout, headers=headers)
    else:
        upload_response = requests.post(upload_url, files=files, timeout=timeout, headers=headers)
    t_upload_end = time.time()
    logger.info("调用上传接口消耗%s（毫秒）" % (round(((t_upload_end - t_upload_start) * 1000), 2)))
    if upload_response.status_code != 200:
        upload_response.raise_for_status()
        raise Exception("调用上传模型到远程临时目录接口失败：{}".format(repr(upload_response)))
    # 如果响应结果是"debuginfo" 字样，有可能是nginx 限制。

    upload_text_response = str_util.to_str(upload_response.text)
    remote_model_path = json_util.to_object(upload_text_response)['field1.path']

    logger.info("上传成功，远程路径为: %s" % remote_model_path)

    return remote_model_path


def _submit2server_custom_model(remote_model_path, repo_name, model_repo_server_portal, timeout, model_format, filename, framework, model_name, description, algorithm_name):
    """
    上传非DCPipeline模型到模型仓库

    Args:
        remote_model_path: 远程服务器上刚才上传的模型的地址。
        repo_name: 仓库名称
        model_repo_server_portal: 模型服务的地址
        timeout: 超时时间
        model_format: 模型格式
        filename: 文件名称
        framework: 模型框架
        model_name: 模型名称
        description: 模型描述
        algorithm_name: 算法名称

    Returns: 模型ID

    """
    dc_proxy = get_dc_proxy()

    submit_url = "%s/aps/mrserver/repo/%s/model/noDCPipeline" % (model_repo_server_portal, repo_name)
    data = json_util.to_json_str({
        "modelPath": remote_model_path,
        "isDCPipeline": "n",
        "modelFileFormat": model_format,
        "fileName": filename,
        "framework": framework,
        "meta": {
            "name": model_name,
            "description": description,
            "algorithmName": algorithm_name,
            "framework": framework
        }
    }
    )
    data = data.encode('utf-8')
    from dc_model_repo.base.mr_log import logger
    logger.info("入库URL:%s" % submit_url)
    logger.info("模型入库参数: %s" % data)
    headers = {'tenantId': dc_proxy.get_tenant_id(), 'projectId': dc_proxy.get_project_id(), 'userId': dc_proxy.get_user_id()}
    submit_response = requests.post(submit_url, data=data, timeout=timeout, headers=headers)
    if submit_response.status_code == 200:
        submit_text_response = str_util.to_str(submit_response.text)
        logger.info("模型入库响应:{}".format(submit_text_response))
        response_dict = json_util.to_object(submit_text_response)

        # 响应的结果样式:
        # {"":0,"data":{"id":"21a67f6c-16bc-4b9f-bfe2-d11eacbc209c"}}

        # 校验响应结果并返回
        code = response_dict.get('code')
        if str(code) != '0':
            raise Exception("调用模型服务入库错误，响应结果: %s" % submit_text_response)
        else:
            return response_dict['data']["id"]
    else:
        submit_response.raise_for_status()
        raise Exception("调用非DCPipeline模型到从远程临时目录到模型仓库接口失败：{}".format(repr(submit_response)))


def _submit2server_pipeline_model(remote_model_path, repo_name, model_repo_server_portal, timeout=TIMEOUT, **kwargs):
    """将远程DCPipeline模型（模型服务可以访问到的地址）提交到模型仓库。

    Args:
        remote_model_path: 远程服务器上刚才上传的模型的地址。
        repo_name: 模型名称。
        model_repo_server_portal: 模型服务的服务地址。
        timeout: 可选 timeout，超时时间。
    Returns: 模型ID
    """
    dc_proxy = get_dc_proxy()

    from dc_model_repo.base.mr_log import logger
    # 1. 调用mr 服务接口入库
    submit_url = "%s/aps/mrserver/repo/%s/model" % (model_repo_server_portal, repo_name)
    data = json_util.to_json_str({"modelPath": remote_model_path})
    logger.info("入库URL: %s" % submit_url)
    logger.info("模型入库参数：%s" % data)
    headers = {'tenantId': dc_proxy.get_tenant_id(), 'projectId': dc_proxy.get_project_id(), 'userId': dc_proxy.get_user_id()}
    submit_response = requests.post(submit_url, data=data, timeout=timeout, headers=headers)
    if submit_response.status_code == 200:
        submit_text_response = str_util.to_str(submit_response.text)
        logger.info("模型入库响应:{}".format(submit_text_response))
        response_dict = json_util.to_object(submit_text_response)

        # 响应的结果样式:
        # {"":0,"data":{"id":"21a67f6c-16bc-4b9f-bfe2-d11eacbc209c"}}

        # 4. 校验响应结果并返回
        code = response_dict.get('code')
        if str(code) != '0':
            raise Exception("调用模型服务入库错误，响应结果:\n%s" % submit_text_response)
        else:
            return response_dict['data']["id"]  # 如果此处报错需要联系MR接口。
    else:
        submit_response.raise_for_status()
        raise Exception("调用非DCPipeline模型到从远程临时目录到模型仓库接口失败：{}".format(repr(submit_response)))


def copy_model(source_repo, target_repo, model_id, model_repo_server_portal, timeout=TIMEOUT):
    """复制模型到模型仓库。

    Args:
        model_repo_server_portal: 模型仓库服务地址。
        source_repo: 源仓库。
        target_repo: 目标仓库。
        model_id: 要复制的模型id。
    Returns:
    """
    from dc_model_repo.base.mr_log import logger
    # 1. 校验参数
    validate_util.require_str_non_empty(model_repo_server_portal, 'model_repo_server_portal')
    validate_util.require_str_non_empty(source_repo, 'source_repo')
    validate_util.require_str_non_empty(target_repo, 'target_repo')
    validate_util.require_str_non_empty(model_id, 'model_id')
    if source_repo == target_repo:
        raise Exception("源仓库和目标仓库不能相同。")

    # 2. 创建请求内容
    data_dict = {
        "sourceRepoName": source_repo,
        "targetRepoName": target_repo,
        "modelIds": model_id
    }
    data = json_util.to_json_str(data_dict)

    # 3. 调用mr服务接口入库
    copy_model_url = "%s/aps/mrserver/repo/model/copy" % model_repo_server_portal
    logger.info("复制模型URL:\n%s\n参数:\n%s" % (copy_model_url, data))

    copy_model_response = requests.post(copy_model_url, data=data, timeout=timeout)
    copy_model_text_response = str_util.to_str(copy_model_response.text)
    logger.info("模型复制响应:")
    logger.info(copy_model_text_response)
    response_dict = json_util.to_object(copy_model_text_response)

    # 响应的结果样式:
    # {
    #     "code": 0,
    #     "data":{
    #         "idMap": {"idOld":"idNew",......}
    #     }
    # }

    # 4. 校验响应结果并返回
    code = response_dict.get('code')
    if str(code) != '0':
        raise Exception("调用模型服务复制错误，响应结果:\n%s" % copy_model_text_response)
    else:
        return response_dict['data']["idMap"][model_id]  # 如果此处报错需要联系MR接口。


def get(model_uri, destination_path, model_repo_server_portal=None, timeout=TIMEOUT):
    """下载模型。

    Args:
        model_uri: 模型URL。
        destination_path: 下载到的本地路径。
        model_repo_server_portal: 模型仓库服务地址。
        timeout: 超时时间。

    Returns:
    """
    dc_proxy = get_dc_proxy()

    import json
    from dc_model_repo.base.mr_log import logger
    model_protocol = 'model://'
    mrserver_export_model_url = '/aps/mrserver/repo/{}/model/{}/export'
    default_repo_name, default_model_repo_server_portal = _get_mr_server_portal()
    # 1. 参数读取校验
    if model_repo_server_portal is None:
        model_repo_server_portal = default_model_repo_server_portal
        if model_repo_server_portal is None:
            raise Exception("参数model_repo_server_portal不能为空。")

    # 解析modelId和reponame
    if model_uri.startswith(model_protocol):
        model_uri = model_uri.replace(model_protocol, '')
        args = model_uri.split('/')
        repo_name = args[0]
        model_id = args[1]
    else:
        raise Exception('解析modelId和repoName失败。')

    # 查询模型下载路径
    request_url = model_repo_server_portal + mrserver_export_model_url.format(repo_name, model_id)
    logger.info(request_url)
    headers = {'tenantId': dc_proxy.get_tenant_id(), 'projectId': dc_proxy.get_project_id(), 'userId': dc_proxy.get_user_id()}
    res = requests.get(request_url, timeout=timeout, headers=headers)
    res.raise_for_status()
    res_text = json.loads(res.text)
    if res_text.get('code') == 0:
        model_download_path = model_repo_server_portal + res_text.get('data').get('path')
        logger.info(model_download_path)
        # 下载模型文件
        logger.info("Start to download model, timeout is {}".format(timeout))
        res = requests.get(model_download_path, timeout=timeout)
        res.raise_for_status()
        with open(destination_path, 'wb') as f:
            for chunk in res.iter_content(100000):
                f.write(chunk)
    else:
        raise Exception(str(res.text))
    logger.info('get model done!')
    return destination_path
