from dc_model_repo.step.userdefined_step import UserDefinedEstimator
from dc_model_repo.base import LearningType, FrameworkType, ModelFileFormatType, Param, ChartData
import os
import shutil
import re
from PIL import Image
import pandas as pd
import cv2
import torchvision
import torch 


class MaskRCNNEstimator(UserDefinedEstimator):

    def __init__(self, input_cols=None, target_cols=None, output_cols=None, **kwargs):
        UserDefinedEstimator.__init__(self=self, input_cols=input_cols, target_cols=target_cols, output_cols=output_cols,
                                      algorithm_name="MaskRCNNSeg",
                                      learning_type=LearningType.Unknown,
                                      framework=FrameworkType.Pytorch,
                                      model_format=ModelFileFormatType.PTH)
                                      
    def get_params(self):
        return None

    def fit(self, X, y=None, **kwargs):
        UserDefinedEstimator.fit(self, X, y, **kwargs)
        
        train_dataset = kwargs["train_dataset"]
        val_dataset = kwargs["val_dataset"]        
        
        train_data_len = train_dataset.length
        test_data_len = val_dataset.length
        
        from model_nets.tv_training_code import train_func
        self.best_params = train_func(self.class_name, self.trainable_backbone_layers, self.step_lr, self.step_weight_decay, self.total_epoch, self.max_trials, self.early_stop, self.tuning_strategy, self.lr, 
                self.batch_size, self.optimizer, self.weight_decay, self.activation_function, self.pretrained_pth, self.device, self.model_dir, self.tensorboard_dir,
                self.performance_path, self.work_dir, train_data_len, test_data_len, self.dataset_type, train_dataset, val_dataset)
        
        params_list = []
        params_list.append(Param(name='tuning_strategy', type='tuning_strategy', value=self.tuning_strategy))
        params_list.append(Param(name='max_trials', type='max_trials', value=self.max_trials))
        params_list.append(Param(name='step_lr', type='step_lr', value=self.step_lr))
        params_list.append(Param(name='step_weight_decay', type='step_weight_decay', value=self.step_weight_decay))
        params_list.append(Param(name='total_epoch', type='total_epoch', value=self.total_epoch))
        params_list.append(Param(name='lr', type='lr', value=self.best_params['lr']))
        params_list.append(Param(name='batch_size', type='batch_size', value=self.best_params['batch_size']))
        params_list.append(Param(name='optimizer', type='optimizer', value=self.best_params['optimizer'])) 
        params_list.append(Param(name='weight_decay', type='weight_decay', value=self.best_params['weight_decay']))        
        params_list.append(Param(name='activation_function', type='activation_function', value=self.best_params['activation_function']))        
        self.params = params_list
        
        return self

    def predict_local(self, X, **kwargs):
        from model_nets import predicting_local
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=len(self.class_name)).to(device)
        model.eval()
        model_dict = torch.load(self.step_path + '/data/model.pth', map_location=device)
        model.load_state_dict(model_dict)
        
        for i in X['image_path']:
            image_name = i.split('/')[-1]
            img = cv2.imread(i)
            predicting_local.predicting(img, model, device, image_name, self.class_name, self.confidence)
        return 0
        
    def predict(self, X, **kwargs):
        from model_nets import predicting
        

        df = pd.DataFrame(columns=['prediction'])
        for i in X['image_path']:
            img = cv2.imread(i)
            predict_res = predicting.predicting(img, self.model, self.device, i, self.class_name, self.confidence)
            df = df.append({'prediction':predict_res},  ignore_index=True)
        return df

    # 保存自定义模型文件到step中
    def persist_model(self, fs, step_path):
        self.step_path = step_path
        # 保存模型的权重文件
        step_data_path = self.serialize_data_path(step_path)
        fs.copy(os.path.join(self.model_dir, 'model.pth'), step_data_path)
        # 保存tensorboard数据
        explanation_path = os.path.join(step_path, 'explanation', 'tensorboard')
        fs.copy(self.tensorboard_dir, explanation_path)
        explanation = [
            ChartData('tensorboard', 'tensorboard', None, {"path": "explanation/tensorboard"})
        ]
        self.explanation = explanation

    # 加载自定义模型
    def prepare(self, step_path, **kwargs):
        self.step_path = step_path
        step_data_path = self.serialize_data_path(step_path)
        
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=False, num_classes=len(self.class_name)).to(device)
        model.eval()
        model_dict = torch.load(self.step_path + '/data/model.pth', map_location=device)
        model.load_state_dict(model_dict)
        self.device = device
        self.model = model

    # def get_persist_step_ignore_variables(self):
    #     return ["device"]

