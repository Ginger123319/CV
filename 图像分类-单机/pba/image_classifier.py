# -*- coding: utf-8 -*-
from dc_model_repo.base.mr_log import logger
import tensorflow as tf
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from dc_model_repo.step.userdefined_step import UserDefinedEstimator
from dc_model_repo.base import DatasetType, LearningType, FrameworkType, ModelFileFormatType, Param, ChartData
from PIL import Image
import shutil


def df_insert_rows(df_origin, df_insert, point):
    return pd.concat([df_origin.iloc[:point, :], df_insert, df_origin.iloc[point:, :]], axis=0).reset_index(drop=True)


def read_rgb_img(img_path):
    try:
        img = Image.open(img_path).convert('RGB')
        if img.width < 1 or img.height < 1:
            print("This image has strange height [{}] or width [{}]. Skip it: {}".format(img.height, img.width, img_path))
            return "Image error: size {}x{}".format(img.height, img.width)
        return img
    except Exception as e:
        print("Error [{}] while reading image. Skip it: {}".format(e, img_path))
        return "Image error: {}".format(str(e))


def write_tfrecord(tfrecord_file, imgs_path, labels, img_col, label_col):
    print("Writing tfrecord: {}".format(tfrecord_file))
    img_cnt = 0
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for img_path, label in zip(imgs_path, labels):
            img = read_rgb_img(img_path)
            if isinstance(img, str):
                continue
            # image_bytes = open(img_path, 'rb').read()
            image_bytes = np.asarray(img).tobytes()
            feature = {img_col: tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes])),
                       label_col: tf.train.Feature(bytes_list=tf.train.BytesList(value=[label.encode("utf-8")])),
                       "shape": tf.train.Feature(int64_list=tf.train.Int64List(value=[img.height, img.width, 3]))}
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            img_cnt += 1
    print("Finished writing. {} skipped, {} kept.".format(len(imgs_path) - img_cnt, img_cnt))


class ImageClassificationEstimator(UserDefinedEstimator):

    # 可选
    def __init__(self, net_config=None, input_cols=None, target_cols=None, output_cols=None):
        assert net_config is not None, "网络配置不能为空！"
        self.net_config = net_config
        from dc_model_repo.base import LearningType
        learning_type = LearningType.BinaryClassify if net_config["class_num"] == 2 else LearningType.MultiClassify
        # 继承父类的属性
        UserDefinedEstimator.__init__(self=self,
                                      input_cols=input_cols,
                                      target_cols=target_cols,
                                      output_cols=output_cols,
                                      learning_type=learning_type,
                                      algorithm_name="IMAGE_CLASSIFICATION_TF2",
                                      output_type=DatasetType.PandasDataFrame,
                                      framework=FrameworkType.TensorFlow2,
                                      model_format=ModelFileFormatType.H5
                                      )

        self.model_type = self.net_config["model_type"]
        self.class_num = self.net_config["class_num"]
        self.width = self.net_config["norm_size"]
        self.height = self.net_config["norm_size"]
        self.input_shape = (self.height, self.width, 3)
        self.optimizer_type = self.net_config["COMPILE_optimizer"]
        self.learning_rate = self.net_config["learning_rate"]
        self.loss_fn_type = self.net_config["COMPILE_loss"]
        self.metrics = [self.net_config["COMPILE_metrics"]]
        self.random_seed = self.net_config["random_seed"]
        self.use_pretrained_weights = self.net_config["model_weights"]  # TODO 这个逻辑需要完善，现在按使用预训练权重
        self.batch_size = self.net_config["FIT_batch_size"]
        self.epochs = self.net_config["FIT_epochs"]
        self.shuffle = self.net_config["FIT_shuffle"]
        # self.image_augment = self.net_config["image_augment"]
        self.weights_dir = self.net_config["weights_dir"]
        self.tensorboard_dir = self.net_config["tensorboard_dir"]
        self.early_stop = self.net_config["early_stop"]

        self.aug_type = self.net_config["augment_type"]
        self.auto_aug_ratio = self.net_config["auto_aug_ratio"]
        self.auto_aug_epochs = self.net_config["auto_aug_epochs"]
        self.image_col = self.net_config["image_col"]
        self.label_col = self.net_config["label_col"]
        self.work_dir = self.net_config["work_dir"]
        self.population_size = self.net_config["population_size"]
        self.auto_aug_config = self.net_config["auto_aug_config"]

        self.tuning_strategy = self.net_config["tuning_strategy"]
        # self.early_stop = self.net_config["early_stop"]
        self.max_trials = self.net_config["max_trials"]
        self.step_lr = self.net_config["step_lr"]
        self.step_weight_decay = self.net_config["step_weight_decay"]

        # self.COMPILE_optimizer = self.net_config["COMPILE_optimizer"]
        # self.FIT_batch_size = self.net_config["FIT_batch_size"]
        # self.learning_rate = self.net_config["learning_rate"]
        self.activation_function = self.net_config["activation_function"]

        self.optimizer_type = self.optimizer_type if len(self.optimizer_type)>1 else self.optimizer_type[0]
        self.batch_size = self.batch_size if len(self.batch_size)>1 else self.batch_size[0]
        self.learning_rate = self.learning_rate if self.learning_rate[0] != self.learning_rate[1] else self.learning_rate[0]
        self.activation_function = self.activation_function if len(self.activation_function)>1 else self.activation_function[0]

        self.need_hypernets = isinstance(self.optimizer_type, list) or isinstance(self.batch_size, list)\
                              or isinstance(self.learning_rate, list) or isinstance(self.activation_function, list)

        self.classes_ = None
        self.model = None
        self.history = None
        self.use_amp = self.net_config["use_amp"]

    # 必选
    def fit(self, X, y=None, image_test=None, label_test=None, **kwargs):
        # 继承父类的属性
        UserDefinedEstimator.fit(self, X, y, **kwargs)

        images_train_path, images_test_path, labels_train_origin, labels_test_origin = X, image_test, y, label_test

        self.classes_ = sorted(list(np.unique(labels_train_origin)))
        assert len(self.classes_) == self.class_num, "Error classes count: Expected {}, actual {}".format(self.class_num, self.classes_)

        train_count = X.shape[0]
        test_count = image_test.shape[0]
        print("Train count: {} Test count: {}".format(train_count, test_count))

        df_train = X.copy()
        df_train[self.label_col] = y
        df_valid = image_test.copy()
        df_valid[self.label_col] = label_test

        df_train_path = os.path.join(self.work_dir, "df_train.csv")
        df_train.to_csv(df_train_path)
        train_tfrecord_file = os.path.join(self.work_dir, "ds_train.tfrecord")
        write_tfrecord(train_tfrecord_file, df_train[self.image_col], df_train[self.label_col], self.image_col, self.label_col)

        df_valid_path = os.path.join(self.work_dir, "df_valid.csv")
        df_valid.to_csv(df_valid_path)
        valid_tfrecord_file = os.path.join(self.work_dir, "ds_valid.tfrecord")
        write_tfrecord(valid_tfrecord_file, df_valid[self.image_col], df_valid[self.label_col], self.image_col, self.label_col)


        train_config = {
            "ds_train_path": train_tfrecord_file,
            "ds_valid_path": valid_tfrecord_file,
            "image_col": self.image_col,
            "label_col": self.label_col,
            "classes": self.classes_,
            "width": self.width,
            "height": self.height,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "shuffle": self.shuffle,
            "random_seed": self.random_seed,
            "model_type": self.model_type,  # VGG16 VGG19 ResNet50 Xception LeNet
            "use_pretrained_weights": self.use_pretrained_weights,
            "weights_dir": self.weights_dir,
            "input_shape": self.input_shape,
            "class_num": self.class_num,
            "optimizer_type": self.optimizer_type,  # SGD RMSprop Adagrad Adadelta Adam Adamax Nadam
            "learning_rate": self.learning_rate,
            "loss_fn_type": self.loss_fn_type,
            "metrics": [self.metrics],
            "early_stop": self.early_stop,
            "tensorboard_dir": self.tensorboard_dir,
            "use_amp": self.use_amp,
        }

        if self.aug_type == "PBA":
            if not self.need_hypernets:
                from .pba_get_schedule import pba_train

                pba_config = train_config.copy()
                pba_config.update({
                    "epochs": self.auto_aug_epochs,
                    "epochs_per_step": 1,
                    "local_dir": self.tensorboard_dir,
                    "population_size": self.population_size,
                    "auto_aug_config": self.auto_aug_config
                })

                if self.auto_aug_ratio >= 1:
                    pba_config["ds_train_path"] = train_tfrecord_file
                    pba_config["ds_valid_path"] = valid_tfrecord_file
                else:
                    df_train_pba, _ = train_test_split(df_train, shuffle=True, stratify=y, train_size=self.auto_aug_ratio, random_state=self.random_seed)
                    df_valid_pba, _ = train_test_split(df_valid, shuffle=True, stratify=label_test, train_size=self.auto_aug_ratio, random_state=self.random_seed)
                    train_count_pba = df_train_pba.shape[0]
                    test_count_pba = df_valid_pba.shape[0]
                    pba_config["batch_count_per_epoch"] = train_count_pba // self.batch_size if train_count_pba % self.batch_size == 0 else (train_count_pba // self.batch_size + 1)
                    pba_config["valid_batch_count_per_epoch"] = test_count_pba // self.batch_size if test_count_pba % self.batch_size == 0 else (test_count_pba // self.batch_size + 1)

                    df_train_pba_path = os.path.join(self.work_dir, "df_train_pba.csv")
                    df_train_pba.to_csv(df_train_pba_path)
                    pba_train_tfrecord_file = os.path.join(self.work_dir, "ds_train_pba.tfrecord")
                    write_tfrecord(pba_train_tfrecord_file, df_train_pba[self.image_col], df_train_pba[self.label_col], self.image_col, self.label_col)

                    df_valid_pba_path = os.path.join(self.work_dir, "df_valid_pba.csv")
                    df_valid_pba.to_csv(df_valid_pba_path)
                    pba_valid_tfrecord_file = os.path.join(self.work_dir, "ds_valid_pba.tfrecord")
                    write_tfrecord(pba_valid_tfrecord_file, df_valid_pba[self.image_col], df_valid_pba[self.label_col], self.image_col, self.label_col)

                    pba_config["ds_train_path"] = pba_train_tfrecord_file
                    pba_config["ds_valid_path"] = pba_valid_tfrecord_file

                schedule = pba_train(pba_config, export_epochs=self.epochs)
            else:
                print("需要进行参数搜索，所以禁用了PBA数据增强，使用随机数据增强。")
                schedule = "Random"
        elif self.aug_type == "Random":
            schedule = "Random"
        elif self.aug_type == "RandAug":
            if self.need_hypernets:
                print("需要进行参数搜索，所以禁用了RandAug数据增强，使用随机数据增强。")
                schedule = "Random"
            else:
                schedule = "RandAug"
        else:
            schedule = None

        from .pba_get_schedule import train_model_with_schedule
        if not self.need_hypernets:
            batch_count_per_epoch = train_count // self.batch_size if train_count % self.batch_size == 0 else (train_count // self.batch_size + 1)
            valid_batch_count_per_epoch = test_count // self.batch_size if test_count % self.batch_size == 0 else (test_count // self.batch_size + 1)

            train_config["batch_count_per_epoch"] = batch_count_per_epoch
            train_config["valid_batch_count_per_epoch"] = valid_batch_count_per_epoch

            self.model, _ = train_model_with_schedule(config=train_config, schedule=schedule)
        else:
            from hypernets.utils.param_tuning import search_params
            from hypernets.core.search_space import Choice, Real
            from hypernets.core import EarlyStoppingCallback

            print("Hyperparameter search space are lr:%s, batch_size:%s, optimizer_type:%s, activation_function:%s" % (
                self.learning_rate, self.batch_size, self.optimizer_type, self.activation_function))
            learning_rate = Real(float(self.learning_rate[0]), float(self.learning_rate[1]), step=self.step_lr) if isinstance(self.learning_rate, list) else self.learning_rate
            batch_size = Choice(self.batch_size) if isinstance(self.batch_size, list) else self.batch_size
            optimizer_type = Choice(self.optimizer_type) if isinstance(self.optimizer_type, list) else self.optimizer_type
            activation_function = Choice(self.activation_function) if isinstance(self.activation_function, list) else self.activation_function
            best_params_ = {'learning_rate': learning_rate, 'batch_size': batch_size, 'optimizer_type': optimizer_type, 'activation_function': activation_function}

            def train_function(learning_rate=learning_rate, batch_size=batch_size, optimizer_type=optimizer_type, activation_function=activation_function):
                train_config["batch_count_per_epoch"] = train_count // batch_size if train_count % batch_size == 0 else (train_count // batch_size + 1)
                train_config["valid_batch_count_per_epoch"] = test_count // batch_size if test_count % batch_size == 0 else (test_count // batch_size + 1)
                train_config["learning_rate"] = learning_rate
                train_config["batch_size"] = batch_size
                train_config["optimizer_type"] = optimizer_type
                train_config["activation_function"] = activation_function
                model, history = train_model_with_schedule(config=train_config, schedule=schedule)

                val_loss = history.history['val_loss'][-1]

                # 当一个trial结束时，保存model，并且更改tensorboard文件的名称
                loss_list = history.history['loss']
                accuracy_list = history.history['accuracy']
                val_loss_list = history.history['val_loss']
                val_accuracy_list = history.history['val_accuracy']

                train_log_file = open(os.path.join(self.work_dir, 'train_log', f'{round(val_loss, 5)}.txt'), 'a')
                for i in range(len(loss_list)):
                    train_log = f'Epoch:{i}/{self.epochs} - loss: {loss_list[i]} - accuracy: {accuracy_list[i]} - val_loss: {val_loss_list[i]} - val_accuracy: {val_accuracy_list[i]}'
                    train_log_file.write(train_log)
                    train_log_file.write('\n')
                train_log_file.close()

                model.save(filepath=os.path.join(self.work_dir, f'{round(val_loss, 5)}.h5'))
                #shutil.move(os.path.join(self.work_dir, 'train'), os.path.join(self.work_dir, f'train{round(val_loss, 5)}'))
                #shutil.move(os.path.join(self.work_dir, 'validation'), os.path.join(self.work_dir, f'validation{round(val_loss, 5)}'))
                print(f"[Trial] reward={round(val_loss, 4)} lr={learning_rate} batch_size={batch_size} optimizer_type={optimizer_type} activation_function={activation_function}")
                return val_loss

            es = EarlyStoppingCallback(max_no_improvement_trials=self.early_stop, mode='min')
            print('Start trial.')

            if os.path.exists(os.path.join(self.work_dir, 'train_log')):
                shutil.rmtree(os.path.join(self.work_dir, 'train_log'))
            os.mkdir(os.path.join(self.work_dir, 'train_log'))
            historys = search_params(func=train_function, searcher=self.tuning_strategy, max_trials=self.max_trials, optimize_direction='min', callbacks=[es])
            best = historys.get_best()
            ps = best.space_sample.get_assigned_params()
            best_params = {p.alias.split('.')[-1]: p.value for p in ps}
            best_params_.update(best_params)
            # 当所有的trial结束后，获取最好的模型, 并且保存tensorboard文件
            #shutil.copytree(os.path.join(self.work_dir, f'train{round(best.reward, 5)}'), os.path.join(self.tensorboard_dir, 'train'))
            #shutil.copytree(os.path.join(self.work_dir, f'validation{round(best.reward, 5)}'), os.path.join(self.tensorboard_dir, 'validation'))
            self.model = tf.keras.models.load_model(filepath=os.path.join(self.work_dir, f'{round(best.reward, 5)}.h5'))
            print("best trial train logs:")
            with open(os.path.join(self.work_dir, 'train_log', f'{round(best.reward, 5)}.txt'), 'r') as f:
                contents = f.readlines()
                for content in contents:
                    print(content)
            print("best_params:", best_params_)
            print('Finished trial.')

            params = [
                Param("learning_rate", None, best_params_['learning_rate']),
                Param("batch_size", None, best_params_['batch_size']),
                Param("optimizer_type", None, best_params_['optimizer_type']),
                Param("activation_function", None, best_params_['activation_function']),
            ]
            assert isinstance(self.params, list)
            self.params.extend(params)

        return self

    # 必选
    def predict(self, X, **kwargs):
        from dc_model_repo.util import dataset_util
        X = dataset_util.validate_and_cast_input_data(X, self.input_type, self.input_features, remove_unnecessary_cols=True)
        old_index = X.index
        X.reset_index(drop=True, inplace=True)
        imgs = []
        skip_index = []
        for i, img_path in enumerate(X[self.input_cols[0]]):
            img = read_rgb_img(img_path)
            if isinstance(img, str):
                skip_index.append((i, img))
                continue
            imgs.append(np.asarray(img.resize((self.width, self.height)), dtype=np.uint8))
        images_input = np.asarray(imgs)
        pred = self.model.predict(images_input)
        prediction = pd.DataFrame({self.output_cols[0]: np.argmax(pred, axis=1), "prob": np.max(pred, axis=1)})
        prediction[self.output_cols[0]] = prediction[self.output_cols[0]].apply(lambda i: self.classes_[i])

        prediction = pd.concat([prediction, pd.DataFrame(pred, columns=["prob_{}".format(c) for c in self.classes_])], axis=1)

        for i, msg in skip_index:
            prediction = df_insert_rows(prediction, pd.DataFrame({self.output_cols[0]: [msg]}), i)
        prediction.index = old_index
        return prediction
        
    #数据处理
    def prepocess(self, X, **kwargs):
        from dc_model_repo.util import dataset_util
        X = dataset_util.validate_and_cast_input_data(X, self.input_type, self.input_features, remove_unnecessary_cols=True)
        old_index = X.index
        X.reset_index(drop=True, inplace=True)
        imgs = []
        skip_index = []
        for i, img_path in enumerate(X[self.input_cols[0]]):
            img = read_rgb_img(img_path)
            if isinstance(img, str):
                skip_index.append((i, img))
                continue
            imgs.append(np.asarray(img.resize((self.width, self.height)), dtype=np.uint8))
        images_input = np.asarray(imgs)
        return images_input

    # 可选
    def persist_model(self, fs, step_path):
        # 保存自定义模型文件到step的data目录
        step_data_path = self.serialize_data_path(step_path)
        model_name = '{}.h5'.format(self.model_type)
        model_data_path = os.path.join(step_data_path, model_name)
        self.model.save(filepath=model_data_path)

        # 保存tensorboard数据
        explanation_path = os.path.join(step_path, 'explanation', 'tensorboard')
        fs.copy(self.tensorboard_dir, explanation_path)
        explanation = [
            ChartData('tensorboard', 'tensorboard', None, {"path": "explanation/tensorboard"}),
            ChartData('netron', 'netron', None, {"path": "data/{}".format(model_name)})
        ]

        self.explanation = explanation

    # 可选
    def prepare(self, step_path, **kwargs):
        # 加载自定义模型
        step_data_path = self.serialize_data_path(step_path)
        model_data_path = os.path.join(step_data_path, '{}.h5'.format(self.model_type))
        self.model = tf.keras.models.load_model(filepath=model_data_path)

    # 可选
    def get_persist_step_ignore_variables(self):
        # 设置保存step时需要忽略的属性
        return ["model", "history"]

    def get_params(self):
        params = [
            Param("model_type", None, self.model_type),
            Param("class_num", None, self.class_num),
            Param("input_shape", None, str(self.input_shape)),
            Param("optimizer_type", None, self.optimizer_type),
            Param("learning_rate", None, self.learning_rate),
            Param("loss_fn_type", None, self.loss_fn_type),
            Param("metrics", None, self.metrics),
            Param("random_seed", None, self.random_seed),
            Param("use_pretrained_weights", None, str(self.use_pretrained_weights)),
            Param("batch_size", None, self.batch_size),
            Param("epochs", None, self.epochs),
            Param("shuffle", None, str(self.shuffle)),
            # Param("image_augment", None, str(self.image_augment)),
            # Param("weights_dir", None, self.weights_dir),
            # Param("tensorboard_dir", None, self.tensorboard_dir),
            Param("early_stop", None, str(self.early_stop))
        ]
        return params
