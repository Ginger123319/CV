import abc
import pandas as pd
from dc_model_repo.pipeline.pipeline import DCPipeline
from dc_model_repo.step.userdefined_step import UserDefinedEstimator


def get_class_str(obj_or_clz):
    import inspect
    clz = obj_or_clz if inspect.isclass(obj_or_clz) is type else type(obj_or_clz)
    return f"{clz.__module__}.{clz.__qualname__}"


def load_class(class_str):
    assert isinstance(class_str, str)
    m, c = class_str.rsplit(".", 1)
    import importlib
    return getattr(importlib.import_module(m), c)


class Estimator(UserDefinedEstimator):
    def __init__(self, model):
        assert isinstance(model, Model)
        from dc_model_repo.base import LearningType
        learning_type = LearningType.Unknown
        # 继承父类的属性
        UserDefinedEstimator.__init__(self=self,
                                      input_cols=["img"],
                                      target_cols=["target"],
                                      output_cols=["pred"],
                                      learning_type=learning_type,
                                      algorithm_name="image_classifier_auto_annotation")
        self.model = model
        self.model_class = get_class_str(model)

    def fit(self, **kwargs):
        # 继承父类的属性
        df = pd.DataFrame({"img": ["a.jpg"], "target": ["t"]})
        m = self.model
        UserDefinedEstimator.fit(self, df.loc[:, ["img"]], df["target"])
        self.model = m
        return self

    def predict(self, X, **kwargs):
        pass

    def persist_model(self, fs, destination):
        # 保存自定义模型文件到step的data目录
        step_data_path = self.serialize_data_path(destination)
        self.model.save_model(step_data_path)
        code_dir = self.serialize_source_code_path(destination)
        import importlib
        from pathlib import Path
        p = Path(importlib.__import__(self.model.__module__).__file__)
        if p.name == "__init__.py":
            source_code = p.parent.resolve()
        else:
            raise Exception("Model is not in a package.")
        import shutil
        # Path(code_dir, source_code.name).mkdir(parents=True, exist_ok=True)
        shutil.copytree(source_code, Path(code_dir, source_code.name))

    def prepare(self, step_path, **kwargs):
        step_data_path = self.serialize_data_path(step_path)
        code = self.serialize_source_code_path(step_path)
        import sys
        sys.path.insert(0, code)
        clz = load_class(self.model_class)
        assert issubclass(clz, Model)
        self.model = clz.load_model(step_data_path)
        self.source_code_path = self.serialize_source_code_path(step_path)

    def get_persist_step_ignore_variables(self):
        return ["model"]


def save_pipeline_model(model, model_id, save_dir):
    assert isinstance(model, Model)
    estimator = Estimator(model)
    estimator.fit()

    steps = [estimator]
    pipeline = DCPipeline(steps=steps,
                          pipeline_id=model_id,
                          name=estimator.algorithm_name,
                          learning_type=estimator.learning_type,
                          input_type=estimator.input_type,
                          input_features=estimator.input_features,
                          sample_data=estimator.sample_data,
                          target_sample_data=estimator.target_sample_data)
    from pathlib import Path
    if Path(save_dir).exists():
        import shutil
        shutil.rmtree(save_dir)
    pipeline.persist(save_dir)
    return pipeline


def load_pipeline_model(model_url, work_dir):
    assert isinstance(model_url, str)
    if model_url.startswith("model://"):
        from pathlib import Path
        model_tmp_path = str(Path(work_dir, "tmp", "pipeline.zip"))
        Path(work_dir, "tmp").mkdir(parents=True, exist_ok=True)
        p_path = str(Path(work_dir, "tmp", "pipeline"))
        from dc_model_repo import model_repo_client
        model_repo_client.get(model_url, model_tmp_path, timeout=(2, 60))

        # 解压模型文件
        def unzip_file(zip_src, dst_dir):
            import zipfile
            r = zipfile.is_zipfile(zip_src)
            if r:
                fz = zipfile.ZipFile(zip_src, 'r')
                for file in fz.namelist():
                    fz.extract(file, dst_dir)
            else:
                print('This is not zip')

        unzip_file(model_tmp_path, p_path)
    else:
        p_path = model_url
    pipeline = DCPipeline.load(p_path)
    pipeline.prepare()

    # Get the model from estimator
    model = pipeline.steps[-1].model
    print("========================================")
    print("Pipeline id: {}".format(pipeline.id))
    print("Class of model: {}".format(type(model)))
    print("========================================")
    return model


class Model(abc.ABC):
    @abc.abstractmethod
    def save_model(self, save_dir):
        pass

    @staticmethod
    @abc.abstractmethod
    def load_model(model_dir):
        pass

    @abc.abstractmethod
    def train_model(self, df_train, df_val, work_dir, is_first_train, **options):
        pass

    @abc.abstractmethod
    def predict(self, df_img, work_dir, **options):
        pass

    @abc.abstractmethod
    def query_hard_example(self, df_img, work_dir, query_cnt=100, strategy="LeastConfidence", **options):
        pass

    @staticmethod
    def split_train_val(df_init, df_anno, is_first_train, val_size=0.2, random_seed=1):
        if val_size > 0:
            from sklearn.model_selection import train_test_split
            df_train, df_val = train_test_split(df_init, test_size=val_size, random_state=random_seed)
        else:
            df_train = df_init
            df_val = None
        if not is_first_train:
            assert df_anno is not None
            df_train = pd.concat([df_train, df_anno], axis=0, ignore_index=True)
        print("Train count: {}\n"
              "Val   count: {}".format(df_train.shape[0], 0 if df_val is None else df_val.shape[0]))
        return df_train, df_val
