
###############################################################
## DCLogger
###############################################################

### write one message to stdout,  one message to stderr
dc.logger.info("Hello DC")
dc.logger.error("This is an error message.")

### write DCConfig object to stdout
dc.logger.info(dc.conf)

### write two messages to stdout, with default seperator
dc.logger.info("Hello", "DC")

### write two messages to stdout, seperate them with ","
dc.logger.info("Hello", "DC", sep=",")

###############################################################
## DCInput,  DCOutput,  DCParam
###############################################################

# access module inputs/outputs/parameters

dc.logger.info(dc.conf.params.classes)
dc.logger.info(dc.conf.params.input_shape)
dc.logger.info(dc.conf.params.val_size)
dc.logger.info(dc.conf.params.lr)
dc.logger.info(dc.conf.params.freeze_epoch)
dc.logger.info(dc.conf.params.total_epoch)
dc.logger.info(dc.conf.params.optimizer)
dc.logger.info(dc.conf.params.batch_size)
dc.logger.info(dc.conf.params.num_of_weights)
dc.logger.info(dc.conf.params.confidence)
dc.logger.info(dc.conf.params.iou)
dc.logger.info(dc.conf.params.use_tfrecord)
dc.logger.info(dc.conf.params.use_amp)
dc.logger.info(dc.conf.params.mosaic)
dc.logger.info(dc.conf.params.test_size)
dc.logger.info(dc.conf.inputs.train_data)
dc.logger.info(dc.conf.inputs.image_data)
dc.logger.info(dc.conf.outputs.performance)
dc.logger.info(dc.conf.outputs.model_dir)
dc.logger.info(dc.conf.outputs.train_logs)
dc.logger.info(dc.conf.outputs.best_model_dir)
dc.logger.info(dc.conf.outputs.prediction)


###############################################################
## DCDataset and DataFrame(Pandas or Spark)
###############################################################

### create a dataset object
ds_train_data = dc.dataset(dc.conf.inputs.train_data)  # from inputs)
ds_image_data = dc.dataset(dc.conf.inputs.image_data)  # from inputs)
ds_performance = dc.dataset(dc.conf.outputs.performance)  # from inputs)
ds_model_dir = dc.dataset(dc.conf.outputs.model_dir)  # from inputs)
ds_train_logs = dc.dataset(dc.conf.outputs.train_logs)  # from inputs)
ds_best_model_dir = dc.dataset(dc.conf.outputs.best_model_dir)  # from inputs)
ds_prediction = dc.dataset(dc.conf.outputs.prediction)  # from inputs)

dc_ds = dc.dataset("ds://my_topic/my_dataset_name")  # from pre-created "DC dataset" in the current project
file_ds = dc.dataset("/path/to/mydata.csv")  # from full file path

### read dataset as DataFrame
df_train_data = ds_train_data.read()
df_image_data = ds_image_data.read()
# or
df_train_data  = dc.dataset(dc.conf.inputs.train_data).read()
df_image_data  = dc.dataset(dc.conf.inputs.image_data).read()


### write DataFrame 'df' to module output
ds_performance = dc.dataset(dc.conf.outputs.performance)
ds_performance.update(df_train_data)
ds_model_dir = dc.dataset(dc.conf.outputs.model_dir)
ds_model_dir.update(df_train_data)
ds_train_logs = dc.dataset(dc.conf.outputs.train_logs)
ds_train_logs.update(df_train_data)
ds_best_model_dir = dc.dataset(dc.conf.outputs.best_model_dir)
ds_best_model_dir.update(df_train_data)
ds_prediction = dc.dataset(dc.conf.outputs.prediction)
ds_prediction.update(df_train_data)
# or
df_train_data.to_dc(dc.conf.outputs.performance)
df_train_data.to_dc(dc.conf.outputs.model_dir)
df_train_data.to_dc(dc.conf.outputs.train_logs)
df_train_data.to_dc(dc.conf.outputs.best_model_dir)
df_train_data.to_dc(dc.conf.outputs.prediction)
# or
df_train_data.to_dc(dc.dataset(performance))
df_train_data.to_dc(dc.dataset(model_dir))
df_train_data.to_dc(dc.dataset(train_logs))
df_train_data.to_dc(dc.dataset(best_model_dir))
df_train_data.to_dc(dc.dataset(prediction))



### update pre-created DC dataset's data with DataFrame 'df'
ds_performance = dc.dataset("ds://my_topic/my_dataset_name")
ds_performance.update(df_train_data)
ds_model_dir = dc.dataset("ds://my_topic/my_dataset_name")
ds_model_dir.update(df_train_data)
ds_train_logs = dc.dataset("ds://my_topic/my_dataset_name")
ds_train_logs.update(df_train_data)
ds_best_model_dir = dc.dataset("ds://my_topic/my_dataset_name")
ds_best_model_dir.update(df_train_data)
ds_prediction = dc.dataset("ds://my_topic/my_dataset_name")
ds_prediction.update(df_train_data)
# or
ds_performance = dc.dataset("ds://my_topic/my_dataset_name")
df_train_data.to_dc(ds_performance)
ds_model_dir = dc.dataset("ds://my_topic/my_dataset_name")
df_train_data.to_dc(ds_model_dir)
ds_train_logs = dc.dataset("ds://my_topic/my_dataset_name")
df_train_data.to_dc(ds_train_logs)
ds_best_model_dir = dc.dataset("ds://my_topic/my_dataset_name")
df_train_data.to_dc(ds_best_model_dir)
ds_prediction = dc.dataset("ds://my_topic/my_dataset_name")
df_train_data.to_dc(ds_prediction)



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



