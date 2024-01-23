from cProfile import label
import numpy as np
from dc_model_repo.step.userdefined_step import UserDefinedEstimator
from dc_model_repo.base import LearningType, FrameworkType, ModelFileFormatType, Param, ChartData
import os
import shutil
import re
from PIL import Image
import pandas as pd



def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # if isinstance(value, type(tf.constant(0))):
    #    value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    import tensorflow as tf
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def read_rgb_img(img_path):
    try:
        img = Image.open(img_path)
        if img.mode != "RGB":
            print("Don't support this image mode: {}. Skip it: {}".format(img.mode, img_path))
            return "Image error: mode {}".format(img.mode)
        if img.width < 1 or img.height < 1:
            print("This image has strange height [{}] or width [{}]. Skip it: {}".format(img.height, img.width, img_path))
            return "Image error: size {}x{}".format(img.height, img.width)
        return img
    except Exception as e:
        print("Error [{}] while reading image. Skip it: {}".format(e, img_path))
        return "Image error: {}".format(str(e))


def parse_sample(annotation_data, input_shape):
    line = annotation_data.split()
    image_path = line[0]
    image = read_rgb_img(image_path)
    if isinstance(image, str):
        return None, None, None

    iw, ih = image.size
    h, w = input_shape
    box = np.array([np.array(list(map(int, box.split(',')))) for box in line[1:]])
    
    # resize image
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dx = (w-nw)//2
    dy = (h-nh)//2

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w,h), (128,128,128))
    new_image.paste(image, (dx, dy))
    image_data = np.asarray(new_image)

    # correct boxes
    box_data = np.zeros((len(box),5))
    if len(box)>0:
        np.random.shuffle(box)
        box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
        box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
        box[:, 0:2][box[:, 0:2]<0] = 0
        box[:, 2][box[:, 2]>w] = w
        box[:, 3][box[:, 3]>h] = h
        box_w = box[:, 2] - box[:, 0]
        box_h = box[:, 3] - box[:, 1]
        box = box[np.logical_and(box_w>1, box_h>1)]
        box_data = np.zeros((len(box),5))
        box_data[:len(box)] = box

    return image_path, image_data, box_data


def write_tfrecord(tfrecord_file, tfrecord_index_file, annotations, input_shape):
    print("Writing tfrecord: {}".format(tfrecord_file))
    filtered_lines = []
    
    img_cnt = 0
    import tensorflow as tf
    import numpy as np
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for anno in annotations:
            image_path, image_data, label = parse_sample(anno, input_shape)
            if image_data is None:
                continue
            feature = {
                "path": _bytes_feature(str(image_path).encode(encoding="utf-8")),
                "image": _bytes_feature(image_data.tobytes()),
                "image_shape": _bytes_feature(str(",".join([str(i) for i in image_data.shape])).encode(encoding="utf-8")),
                "label": _bytes_feature(label.tobytes()),
                "label_shape": _bytes_feature(str(",".join([str(i) for i in label.shape])).encode(encoding="utf-8"))
            }
            tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(tf_example.SerializeToString())
            filtered_lines.append(anno)
            img_cnt += 1
    print("Finished writing. {} skipped, {} kept.".format(len(annotations) - img_cnt, img_cnt))

    # 创建索引 TODO
    from .utils import create_index
    print("Creating index file: {}".format(tfrecord_index_file))
    create_index.create_index(tfrecord_file, tfrecord_index_file)
    print("Index file created!")
    return filtered_lines



class SSDEstimator(UserDefinedEstimator):

    def __init__(self, input_cols=None, target_cols=None, output_cols=None, **kwargs):

        UserDefinedEstimator.__init__(self=self, input_cols=input_cols, target_cols=target_cols,
                                      output_cols=output_cols,
                                      algorithm_name="SSD",
                                      learning_type=LearningType.Unknown,
                                      framework=FrameworkType.Pytorch,
                                      model_format=ModelFileFormatType.PTH)

    def get_params(self):

        params_list = []
        params_list.append(Param(name='image_size', type='image_size', value=self.image_size))
        params_list.append(Param(name='lr', type='lr', value=self.lr))
        params_list.append(Param(name='freeze_epoch', type='freeze_epoch', value=self.freeze_epoch))
        params_list.append(Param(name='total_epoch', type='total_epoch', value=self.total_epoch))
        params_list.append(Param(name='optimizer', type='optimizer', value=self.optimizer))
        params_list.append(Param(name='batch_size', type='batch_size', value=self.batch_size))
        params_list.append(Param(name='num_classes', type='num_classes', value=self.num_classes))

        return params_list

    def fit(self, X, y, **kwargs):

        UserDefinedEstimator.fit(self, X, y, **kwargs)

            
        with open(self.annotation_path) as f:
            lines = f.readlines()
        self.num_val = int(len(lines) * self.val_size)
        self.num_train = len(lines) - self.num_val
                
        if self.use_tfrecord:
            self.train_tf_record_file = os.path.join(self.work_dir, "middle_dir/tfrecord_data/train.tfrecord")
            self.train_tf_record_index = os.path.join(self.work_dir, "middle_dir/tfrecord_data/train.tfrecord.index")
            filtered_lines_train = write_tfrecord(tfrecord_file=self.train_tf_record_file,
                                                  tfrecord_index_file=self.train_tf_record_index,
                                                  annotations=lines[:self.num_train],
                                                  input_shape=(self.image_size, self.image_size))
            self.num_train = len(filtered_lines_train)

            self.val_tf_record_file = os.path.join(self.work_dir, "middle_dir/tfrecord_data/val.tfrecord")
            self.val_tf_record_index = os.path.join(self.work_dir, "middle_dir/tfrecord_data/val.tfrecord.index")
            filtered_lines_val = write_tfrecord(tfrecord_file=self.val_tf_record_file,
                                                tfrecord_index_file=self.val_tf_record_index,
                                                annotations=lines[self.num_train:],
                                                input_shape=(self.image_size, self.image_size))
            self.num_val = len(filtered_lines_val)



        if self.image_augment == "PBA":         

            # TODO 在这里添加PBT训练, 获取schedule
            from model_nets.pba import pba_get_schedule
            train_config = {"num_classes": self.num_classes+1,
                            "image_size": self.image_size,
                            "val_size": self.val_size,
                            "work_dir": self.work_dir,
                            "tensorboard_dir": self.tensorboard_dir,
                            "use_tfrecord": self.use_tfrecord,
                            "use_amp": self.use_amp,
                            "lr": self.lr,
                            "freeze_epoch": self.freeze_epoch,
                            "total_epoch": self.total_epoch,
                            "optimizer": self.optimizer,
                            "batch_size": self.batch_size,
                            "population_size": self.population_size,
                            "local_dir": self.tensorboard_dir,
                            "epochs":self.auto_aug_epochs,
                            "min_dim":self.image_size,
                            "model_path":self.work_dir + "/pre_training_weights/pre_training_weights.pth",
                            "annotation_path":self.work_dir+'/middle_dir/data_info/pba_info.txt'}
                        
            if self.use_tfrecord:
                with open(self.work_dir+'/middle_dir/data_info/pba_info.txt') as f:
                    lines_pba = f.readlines()
                num_val_pba = int(len(lines_pba) * self.val_size)
                num_train_pba = len(lines_pba) - num_val_pba

                train_tf_record_file_pba = os.path.join(self.work_dir, "middle_dir/tfrecord_data/train_pba.tfrecord")
                train_tf_record_index_pba = os.path.join(self.work_dir, "middle_dir/tfrecord_data/train_pba.tfrecord.index")
                filtered_lines_train_pba = write_tfrecord(tfrecord_file=train_tf_record_file_pba,
                            tfrecord_index_file=train_tf_record_index_pba,
                            annotations=lines_pba[:num_train_pba],
                            input_shape=(self.image_size, self.image_size))
                num_train_pba = len(filtered_lines_train_pba)
                val_tf_record_file_pba = os.path.join(self.work_dir, "middle_dir/tfrecord_data/val_pba.tfrecord")
                val_tf_record_index_pba = os.path.join(self.work_dir, "middle_dir/tfrecord_data/val_pba.tfrecord.index")
                filtered_lines_val_pba = write_tfrecord(tfrecord_file=val_tf_record_file_pba,
                            tfrecord_index_file=val_tf_record_index_pba,
                            annotations=lines_pba[num_train_pba:],
                            input_shape=(self.image_size, self.image_size))   
                num_val_pba = len(filtered_lines_val_pba)
                train_config["num_train_pba"] = num_train_pba
                train_config["num_val_pba"] = num_val_pba
                train_config["train_tf_record_file_pba"] = train_tf_record_file_pba
                train_config["train_tf_record_index_pba"] = train_tf_record_index_pba
                train_config["val_tf_record_file_pba"] = val_tf_record_file_pba
                train_config["val_tf_record_index_pba"] = val_tf_record_index_pba
            schedule = pba_get_schedule.pba_train(train_config=train_config, export_epochs=self.total_epoch)
        elif self.image_augment=="Random":
            schedule = "Random"
        elif self.image_augment=="Disable":
            schedule = None
        elif self.image_augment=="RandAug":
            schedule = "RandAug"
        else:
            raise Exception("Unexpected value for image_augment: {}".format(self.image_augment))

        # 下面的训练中添加对随机增强、不增强、按照策略增强等方式
        if self.image_augment=="RandAug":
            # do randaug here.
            def train_one_trail(aug_config):
                from model_nets.training import Train

                training = Train(self.num_classes, self.image_size, self.num_train, self.num_val, lines, self.work_dir, self.tensorboard_dir, self.use_tfrecord, self.use_amp, mosaic=self.mosaic)
                training.train(self.lr, self.freeze_epoch, self.total_epoch, self.optimizer, self.batch_size, aug_config)
            n_m = list()
            for p_n in [1, 2]:
                for p_m in [v/10 for v in range(0,11)]:
                    n_m.append((p_n, p_m))

            import multiprocessing
            for i, (p_n, p_m) in enumerate(n_m):
                print(f"Start RandAug trail {i+1}/{len(n_m)}: N {p_n} M {p_m} ...")
                process_eval = multiprocessing.Process(target=train_one_trail, args=({"N":p_n, "M":p_m},))
                process_eval.start()
                process_eval.join()
                print(f"End RandAug trail {i+1}/{len(n_m)}: N {p_n} M {p_m}.")
        else:
            from model_nets.training import Train

            training = Train(self.num_classes, self.image_size, self.num_train, self.num_val, lines, self.work_dir, self.tensorboard_dir, self.use_tfrecord, self.use_amp, mosaic=self.mosaic)
            training.train(self.lr, self.freeze_epoch, self.total_epoch, self.optimizer, self.batch_size, schedule)

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

        from model_nets.ssd import SSD

        ssd = SSD(self.image_size, self.step_path)
        df = pd.DataFrame(columns=['prediction'])

        for i in range(len(X)):
            image_path = X['path'][i]
            image = Image.open(image_path)
            image = image.convert("RGB")
            image_info_list = ssd.detect_image(image)
            df = df.append({'prediction': str(image_info_list)}, ignore_index=True)

        return df

    def predict2(self, X, **kwargs):

        from model_nets.ssd2 import SSD

        ssd = SSD(self.image_size, self.work_dir)
        df = pd.DataFrame(columns=['prediction'])

        for i in range(len(X)):
            image_path = X['path'][i]
            image = Image.open(image_path)
            image = image.convert("RGB")
            image_info_list = ssd.detect_image(image)
            df = df.append({'prediction': str(image_info_list)}, ignore_index=True)

        return df

    # 保存自定义模型文件到step的data目录
    def persist_model(self, fs, step_path):

        step_data_path = self.serialize_data_path(step_path)
        fs.copy(self.work_dir + '/middle_dir/data_info/classes.txt', step_data_path)
        fs.copy(self.work_dir + '/middle_dir/logs/' + self.best_model_pth, step_data_path)

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

    def get_persist_step_ignore_variables(self):

        return ["model"]
