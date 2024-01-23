from dc_model_repo.step.userdefined_step import UserDefinedEstimator
from dc_model_repo.base import LearningType, FrameworkType, ModelFileFormatType

import os
import shutil
import re
from PIL import Image
import pandas as pd


class YoloV4Estimator(UserDefinedEstimator):

    def __init__(self, input_cols=None, target_cols=None, output_cols=None, **kwargs):

        UserDefinedEstimator.__init__(self=self, input_cols=input_cols, target_cols=target_cols,
                                      output_cols=output_cols,
                                      algorithm_name="YoloV4",
                                      learning_type=LearningType.Unknown,
                                      framework=FrameworkType.Pytorch,
                                      model_format=ModelFileFormatType.PTH)

    def get_params(self):

        from dc_model_repo.base import Param

        params_list = []
        params_list.append(Param(name='image_size', type='image_size', value=self.image_size))
        params_list.append(Param(name='lr', type='lr', value=self.lr))
        params_list.append(Param(name='freeze_epoch', type='freeze_epoch', value=self.freeze_epoch))
        params_list.append(Param(name='total_epoch', type='total_epoch', value=self.total_epoch))
        params_list.append(Param(name='optimizer', type='optimizer', value=self.optimizer))
        params_list.append(Param(name='batch_size', type='batch_size', value=self.batch_size))
        params_list.append(Param(name='smooth_label', type='smooth_label', value=self.smooth_label))
        params_list.append(Param(name='mosaic', type='mosaic', value=self.mosaic))
        params_list.append(Param(name='Cosine_lr', type='Cosine_lr', value=self.Cosine_lr))
        params_list.append(Param(name='num_classes', type='num_classes', value=self.num_classes))

        return params_list

    def fit(self, X, y, **kwargs):

        UserDefinedEstimator.fit(self, X, y, **kwargs)

        from model_nets.training import Train
        training = Train(self.num_classes, self.image_size, self.val_size, self.Cosine_lr, self.mosaic,
                         self.smooth_label, self.work_dir, self.tensorboard_dir, self.use_tfrecord, self.use_amp)
        training.train(self.lr, self.freeze_epoch, self.total_epoch, self.optimizer, self.batch_size)

        # 筛选出val loss最低的一个pth文件
        logs_list_dir = os.listdir(self.work_dir + '/middle_dir/logs')
        val_dict = {}  # 组成{pth文件名：val loss值}形式的字典
        val_list = []  # 组成[val loss值]形式的列表
        for i in logs_list_dir:
            if i.endswith('.pth'):
                val_loss = float(re.match(r'Epoch(.+)-Total_Loss(.+)-Val_Loss(.+).pth', i).group(3))
                val_dict[i] = val_loss
                val_list.append(val_loss)

        val_list.sort()

        val_list_n = val_list[:self.n_weights_saved]  # 筛选出val loss从小到大排列的前n名
        val_key_list = []  # 将val loss排前n名的pth文件名保存在列表里
        for i in val_dict.keys():
            if val_dict[i] in val_list_n:
                val_key_list.append(i)
        for i in val_key_list:
            os.system('cp %s %s' % (self.work_dir + '/middle_dir/logs/' + i, self.pth_logs_dir))

        val_list_1 = val_list_n[:1]  # 筛选出val loss从小到大排列的第一名
        val_key_list_1 = []  # 将val loss排第一名的pth文件名保存在列表里
        for i in val_dict.keys():
            if val_dict[i] in val_list_1:
                val_key_list_1.append(i)
                break

        for i in val_key_list_1:
            os.system('cp %s %s' % (
            self.work_dir + '/middle_dir/logs/' + i, self.work_dir + '/middle_dir/normal_train_best_model_dir'))

        self.best_model_pth = val_key_list_1[0]

        return self

    def predict(self, X, **kwargs):

        from model_nets.yolo import YOLO

        yolo = YOLO(self.image_size, self.step_path)
        df = pd.DataFrame(columns=['prediction'])

        for i in range(len(X)):
            image_path = X['path'][i]
            image = Image.open(image_path)
            image = image.convert("RGB")
            image_info_list = yolo.detect_image(image)
            df = df.append({'prediction': str(image_info_list)}, ignore_index=True)

        return df

    def predict2(self, X, **kwargs):

        from model_nets.yolo2 import YOLO

        yolo = YOLO(self.image_size, self.work_dir)
        df = pd.DataFrame(columns=['prediction'])

        for i in range(len(X)):
            image_path = X['path'][i]
            image = Image.open(image_path)
            image = image.convert("RGB")
            image_info_list = yolo.detect_image(image)
            df = df.append({'prediction': str(image_info_list)}, ignore_index=True)

        return df

    # 可选
    def persist_model(self, fs, step_path):
        # 保存自定义模型文件到step的data目录
        step_data_path = self.serialize_data_path(step_path)
        fs.copy(self.work_dir + '/middle_dir/data_info/classes.txt', step_data_path)
        fs.copy('script_files/yolo_anchors.txt', step_data_path)
        fs.copy(self.work_dir + '/middle_dir/logs/' + self.best_model_pth, step_data_path)

        # 保存tensorboard数据
        from dc_model_repo.base import ChartData

        explanation_path = os.path.join(step_path, 'explanation', 'tensorboard')
        fs.copy(self.tensorboard_dir, explanation_path)
        explanation = [
            ChartData('tensorboard', 'tensorboard', None, {"path": "explanation/tensorboard"})
        ]
        self.explanation = explanation

        # 可选

    def prepare(self, step_path, **kwargs):
        self.step_path = step_path
        # 加载自定义模型
        step_data_path = self.serialize_data_path(step_path)

    def get_persist_step_ignore_variables(self):

        return ["model"]
