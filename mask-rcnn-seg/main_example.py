
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

dc.logger.info(dc.conf.params.class_name)
dc.logger.info(dc.conf.params.trainable_backbone_layers)
dc.logger.info(dc.conf.params.pretrained_backbone)
dc.logger.info(dc.conf.params.confidence)
dc.logger.info(dc.conf.params.step_lr)
dc.logger.info(dc.conf.params.step_weight_decay)
dc.logger.info(dc.conf.params.total_epoch)
dc.logger.info(dc.conf.params.max_trials)
dc.logger.info(dc.conf.params.early_stop)
dc.logger.info(dc.conf.params.tuning_strategy)
dc.logger.info(dc.conf.params.lr)
dc.logger.info(dc.conf.params.batch_size)
dc.logger.info(dc.conf.params.optimizer)
dc.logger.info(dc.conf.params.weight_decay)
dc.logger.info(dc.conf.params.activation_function)
dc.logger.info(dc.conf.params.val_ratio)
dc.logger.info(dc.conf.inputs.train_dataset)
dc.logger.info(dc.conf.inputs.instance_segmentation_dataset)
dc.logger.info(dc.conf.outputs.prediction)
dc.logger.info(dc.conf.outputs.model_dir)
dc.logger.info(dc.conf.outputs.train_logs_dir)
dc.logger.info(dc.conf.outputs.performance)


###############################################################
## DCDataset and DataFrame(Pandas or Spark)
###############################################################

### create a dataset object
ds_train_dataset = dc.dataset(dc.conf.inputs.train_dataset)  # from inputs)
ds_instance_segmentation_dataset = dc.dataset(dc.conf.inputs.instance_segmentation_dataset)  # from inputs)
ds_prediction = dc.dataset(dc.conf.outputs.prediction)  # from inputs)
ds_model_dir = dc.dataset(dc.conf.outputs.model_dir)  # from inputs)
ds_train_logs_dir = dc.dataset(dc.conf.outputs.train_logs_dir)  # from inputs)
ds_performance = dc.dataset(dc.conf.outputs.performance)  # from inputs)

ds = dc.dataset("ds://my_topic/my_dataset_name")  # from pre-created "dataset" in the current project
file_ds = dc.dataset("/path/to/mydata.csv")  # from full file path

### read dataset as DataFrame
df_train_dataset = ds_train_dataset.read()
df_instance_segmentation_dataset = ds_instance_segmentation_dataset.read()
# or
df_train_dataset  = dc.dataset(dc.conf.inputs.train_dataset).read()
df_instance_segmentation_dataset  = dc.dataset(dc.conf.inputs.instance_segmentation_dataset).read()


### write DataFrame 'df' to module output
ds_prediction = dc.dataset(dc.conf.outputs.prediction)
ds_prediction.update(df_train_dataset)
ds_model_dir = dc.dataset(dc.conf.outputs.model_dir)
ds_model_dir.update(df_train_dataset)
ds_train_logs_dir = dc.dataset(dc.conf.outputs.train_logs_dir)
ds_train_logs_dir.update(df_train_dataset)
ds_performance = dc.dataset(dc.conf.outputs.performance)
ds_performance.update(df_train_dataset)
# or
df_train_dataset.to_dc(dc.conf.outputs.prediction)
df_train_dataset.to_dc(dc.conf.outputs.model_dir)
df_train_dataset.to_dc(dc.conf.outputs.train_logs_dir)
df_train_dataset.to_dc(dc.conf.outputs.performance)
# or
df_train_dataset.to_dc(dc.dataset(prediction))
df_train_dataset.to_dc(dc.dataset(model_dir))
df_train_dataset.to_dc(dc.dataset(train_logs_dir))
df_train_dataset.to_dc(dc.dataset(performance))



### update pre-created dataset's data with DataFrame 'df'
ds_prediction = dc.dataset("ds://my_topic/my_dataset_name")
ds_prediction.update(df_train_dataset)
ds_model_dir = dc.dataset("ds://my_topic/my_dataset_name")
ds_model_dir.update(df_train_dataset)
ds_train_logs_dir = dc.dataset("ds://my_topic/my_dataset_name")
ds_train_logs_dir.update(df_train_dataset)
ds_performance = dc.dataset("ds://my_topic/my_dataset_name")
ds_performance.update(df_train_dataset)
# or
ds_prediction = dc.dataset("ds://my_topic/my_dataset_name")
df_train_dataset.to_dc(ds_prediction)
ds_model_dir = dc.dataset("ds://my_topic/my_dataset_name")
df_train_dataset.to_dc(ds_model_dir)
ds_train_logs_dir = dc.dataset("ds://my_topic/my_dataset_name")
df_train_dataset.to_dc(ds_train_logs_dir)
ds_performance = dc.dataset("ds://my_topic/my_dataset_name")
df_train_dataset.to_dc(ds_performance)



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



