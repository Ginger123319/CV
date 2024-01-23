
###############################################################
## DCLogger
###############################################################

### write one message to stdout,  one message to stderr
dc.logger.info("Hello World")
dc.logger.error("This is an error message.")

### write DCConfig object to stdout
dc.logger.info(dc.conf)

### write two messages to stdout, with default seperator
dc.logger.info("Hello", "World")

### write two messages to stdout, seperate them with ","
dc.logger.info("Hello", "World", sep=",")

###############################################################
## DCInput,  DCOutput,  DCParam
###############################################################

# access module inputs/outputs/parameters

dc.logger.info(dc.conf.params.val_size)
dc.logger.info(dc.conf.params.query_cnt)
dc.logger.info(dc.conf.params.strategy)
dc.logger.info(dc.conf.params.model_url)
dc.logger.info(dc.conf.params.input_label_path)
dc.logger.info(dc.conf.params.input_unlabel_path)
dc.logger.info(dc.conf.params.input_added_label_path)
dc.logger.info(dc.conf.params.target_model_Id)
dc.logger.info(dc.conf.params.target_model_path)
dc.logger.info(dc.conf.params.is_first_train)
dc.logger.info(dc.conf.params.label_type)
dc.logger.info(dc.conf.params.result_path)
dc.logger.info(dc.conf.params.options)
dc.logger.info(dc.conf.params.partition_dir)


###############################################################
## DCDataset and DataFrame(Pandas or Spark)
###############################################################

### create a dataset object

ds = dc.dataset("ds://my_topic/my_dataset_name")  # from pre-created "dataset" in the current project
file_ds = dc.dataset("/path/to/mydata.csv")  # from full file path

### read dataset as DataFrame


### write DataFrame 'df' to module output



### update pre-created dataset's data with DataFrame 'df'



# 自定义Transfromer代码示例，推荐在一个新的py文件中实现。
from dc_model_repo.step.base import UserDefinedDCStep

class ExampleTransformer(UserDefinedDCStep):

    def get_params(self):
        return None

    def persist_model(self, fs, destination):
        # 如果模型有附件需要实现此方法
        import os
        super(BertCustomEstimator, self).persist_model(fs, destination)
        model_path = os.path.join(self.serialize_data_path(destination), 'model')
        shutil.copytree(self.attachments_path, model_path)

    def fit(self, X, y=None, **kwargs):
        super(ExampleTransformer, self).fit(X, y, **kwargs)

    def transform(self, X):
        return X

    def prepare(self, step_path, **kwargs):
        # 可以在此处加载模型，更优于在transform方法中加载模型
        pass


# 自定义Estimator代码示例，推荐在一个新的py文件中实现。
from dc_model_repo.step.base import UserDefinedDCStep, BaseEstimator

class ExampleEstimator(UserDefinedDCStep, BaseEstimator):

    def get_params(self):
        return None

    def persist_model(self, fs, destination):
        # 如果模型有附件需要实现此方法
        import os
        super(BertCustomEstimator, self).persist_model(fs, destination)
        model_path = os.path.join(self.serialize_data_path(destination), 'model')
        shutil.copytree(self.attachments_path, model_path)

    def fit(self, X, y=None, **kwargs):
        super(ExampleTransformer, self).fit(X, y, **kwargs)

    def transform(self, X):
        return X

    def prepare(self, step_path, **kwargs):
        # 可以在此处加载模型，更优于在predict方法中加载模型
        pass

    def predict(self, X):
        # 如果为Estimaor需要重写此方法
        self.transform(X)



