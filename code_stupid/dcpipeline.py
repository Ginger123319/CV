import pandas as pd
from sklearn import model_selection

from dc_model_repo import model_repo
from dc_model_repo.base import ChartData

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 9999)

train_data_path = "../datasets/titanic/train.csv"

df = pd.read_csv(train_data_path)

# 删除无效列
X_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"]
y_column = "Survived"
# 取出对应的列，第一个位置代表行，第二个位置代表列
X = df.loc[:, X_columns]
y = df[y_column]

# 拆分数据集
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3, stratify=y)
X_train_backup = X_train.copy(deep=True)

# 初始化pipeline
from dc_model_repo.step.aps_step import PipelineInitDCStep

step_init = PipelineInitDCStep(label_col=y_column)
step_init.fit(X_train, y_train)
X_train = step_init.transform(X_train)


# 入模型训练
from custom_estimator.classification import MyClassifier

step_model = MyClassifier()
step_model.fit(X_train, y_train)
prediction = step_model.predict(X_train)

# 获取在训练集上的指标数据
from sklearn import metrics

auc_on_train = metrics.roc_auc_score(y_train, prediction.iloc[:, 1])
print("=============== AUC on train data:", auc_on_train)

# 组装pipeline，并对训练集进行预测
from dc_model_repo.pipeline.pipeline import DCPipeline
from dc_model_repo.base import LearningType
from dc_model_repo.base.meta_data import Performance

pipeline = DCPipeline(steps=[step_model],
                      name='Titanic_pipeline',
                      learning_type=LearningType.Unknown,
                      input_type=step_init.input_type,
                      input_features=step_init.input_features,
                      sample_data=step_init.sample_data,
                      target_sample_data=step_init.target_sample_data,
                      performance=[ChartData('metrics', Performance.Metrics, {"auc": auc_on_train})])


persist_path = "./model_pipeline/titanic_v2"
pipeline.persist(persist_path)

from dc_model_repo import model_repo_client
model_repo_client.submit(persist_path)