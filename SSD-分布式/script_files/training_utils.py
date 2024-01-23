import os
import shutil


def make_middle_dir(work_dir):
    # 各文件夹的功能说明
    # middle_dir_ssd_dis 用于保存训练过程中生成的中间结果
    # data_info 用于保存数据集的路径，标注框，类别等信息
    # logs 用于保存训练过程中生成的pth文件
    # tensorboard_logs 用于保存训练时产生的tensorboard日志
    # model_dir 用于保存val loss较低的n个pth文件
    # normal_train_best_model_dir 用于保存正常训练中val_loss最低的1个pth文件
    # image_info 用于保存模型评估时ground_truth和detect_result的结果
    # evaluate_result 用于保存模型评估结果

    if os.path.exists(work_dir + "/middle_dir_ssd_dis"):
        shutil.rmtree(work_dir + "/middle_dir_ssd_dis")

    os.makedirs(work_dir + "/middle_dir_ssd_dis")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/data_info")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/logs")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/tensorboard_logs")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/model_dir")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/normal_train_best_model_dir")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/tfrecord_data")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/image_info")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/image_info/ground_truth")
    os.makedirs(work_dir + "/middle_dir_ssd_dis/evaluate_result")

    print("middle_dir_ssd_dis has been created!")


def destroy_middle_dir(work_dir):
    shutil.rmtree(work_dir + "/middle_dir_ssd_dis", ignore_errors=True)
    print("middle_dir_ssd_dis has been destroyed!")


def create_train_info_txt(train_data, classes, work_dir):
    # 生成train_info.txt, 保存图像的路径和标注信息
    f = open(work_dir + '/middle_dir_ssd_dis/data_info/train_info.txt', 'w')
    for i in range(len(train_data)):
        path = train_data.loc[i].path
        f.write(path)
        for j in train_data.loc[i].target:
            f.write(
                ' ' + str(j['left']) + "," + str(j['top']) + "," + str(j['right']) + "," + str(j['bottom']) + "," + str(
                    classes.index(j['label'])))
        f.write('\n')
    f.close()

    print("train_info.txt has been created.")


def create_classes_txt(classes, work_dir):
    # 生成class.txt文件，用于记录目标检测的类别
    f = open(work_dir + '/middle_dir_ssd_dis/data_info/classes.txt', 'w')
    for i in classes:
        f.write(i)
        f.write('\n')
    f.close()
    print("classes.txt has been created.")


def load_pre_training_weights(pre_training_weights_path, work_dir):
    if os.path.exists(work_dir + "/pre_training_weights_ssd"):
        shutil.rmtree(work_dir + "/pre_training_weights_ssd")
    os.makedirs(work_dir + "/pre_training_weights_ssd")

    shutil.copyfile(pre_training_weights_path, work_dir + '/pre_training_weights_ssd/pre_training_weights.pth')
    print("pre_training_weights_ssd has been loaded!")
