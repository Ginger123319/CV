import shutil

import pandas
import tensorflow as tf
import os


def tfrecord_split(source_path: str, dest_path: str, x_train: pandas.DataFrame, x_test: pandas.DataFrame, path_column: str, strategy: str):
    """
    对tfrecord文件进行训练集和测试集拆分，拆分方式有：顺序拆分和随机拆分;
    其中tfrecord文件名格式为：<no>_data_<样本数>.tfrecord
    :param source_path: tfrecord文件路径，file or dir
    :param dest_path: 拆分后的路径，会生成两个目录：train、test
    :param x_train: 训练集dataframe
    :param x_test: 测试集dataframe
    :param path_column: csv文件中样本路径的列名
    :param strategy: 拆分策略：head、random
    :return:
        train_dest_path, test_dest_path
    :exception
        split error
    """
    assert os.path.exists(source_path), f"{source_path} is no such file or directory"
    assert strategy in ('head', 'random'), f"strategy {strategy} is invalid"

    train_tfrecord_dir = os.path.join(dest_path, 'instance_segmentation/train')
    test_tfrecord_dir = os.path.join(dest_path, 'instance_segmentation/test')
    if not os.path.exists(train_tfrecord_dir): os.makedirs(train_tfrecord_dir)
    if not os.path.exists(test_tfrecord_dir): os.makedirs(test_tfrecord_dir)

    train_total_sample = x_train.shape[0]
    test_total_sample = x_test.shape[0]

    image_feature_description = {
        path_column: tf.io.FixedLenFeature([], tf.string)
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        return tf.io.parse_single_example(example_proto, image_feature_description), example_proto

    # 处理单tfrecord文件
    if os.path.isfile(source_path):
        raw_dataset = tf.data.TFRecordDataset(source_path)
        train_tfrecord_file = os.path.join(train_tfrecord_dir, f'0_data_{train_total_sample}.tfrecord')
        test_tfrecord_file = os.path.join(test_tfrecord_dir, f'0_data_{test_total_sample}.tfrecord')

        if strategy == 'head':
            with tf.data.experimental.TFRecordWriter(test_tfrecord_file) as writer_test:
                writer_test.write(raw_dataset.take(test_total_sample))

            with tf.data.experimental.TFRecordWriter(train_tfrecord_file) as writer_train:
                writer_train.write(raw_dataset.skip(test_total_sample))

        elif strategy == 'random':
            train_sample = x_train[path_column].tolist()
            test_sample = x_test[path_column].tolist()

            with tf.io.TFRecordWriter(test_tfrecord_file) as writer_test, tf.io.TFRecordWriter(train_tfrecord_file) as writer_train:
                parsed_dataset = raw_dataset.map(_parse_image_function)
                for t1, t2 in parsed_dataset:
                    file_path = tf.compat.as_str_any(t1[path_column].numpy())
                    if file_path in train_sample:
                        writer_train.write(tf.compat.as_bytes(t2.numpy()))
                    if file_path in test_sample:
                        writer_test.write(tf.compat.as_bytes(t2.numpy()))

    # 处理tfrecord目录
    if os.path.isdir(source_path):
        # 组织所有tfrecord文件顺序
        all_tfrecord_file = []
        for f in os.listdir(source_path):
            abs_f = os.path.join(source_path, f)
            if os.path.isdir(abs_f) or (os.path.isfile(abs_f) and not abs_f.endswith(".tfrecord")):
                continue
            f_no = int(os.path.basename(abs_f).split('_')[0])
            all_tfrecord_file.insert(f_no, abs_f)

        if strategy == 'head':
            # 遍历所有tfrecord文件，拆分训练、测试集
            view_sample_num = 0
            file_no = 0
            for abs_f in all_tfrecord_file:
                one_f_sample_num = int(os.path.basename(abs_f).split('_')[-1].split('.')[0])
                view_sample_num += one_f_sample_num
                if view_sample_num < test_total_sample:
                    shutil.copyfile(abs_f, os.path.join(test_tfrecord_dir, os.path.basename(abs_f)))
                elif view_sample_num == test_total_sample:
                    shutil.copyfile(abs_f, os.path.join(test_tfrecord_dir, os.path.basename(abs_f)))
                    for _abs_f in all_tfrecord_file[file_no + 1:]:
                        shutil.copyfile(_abs_f, os.path.join(train_tfrecord_dir, os.path.basename(_abs_f)))
                    break
                else:
                    one_f_train_sample_num = view_sample_num - test_total_sample
                    one_f_test_sample_num = one_f_sample_num - (view_sample_num - test_total_sample)
                    train_tfrecord_file = os.path.join(train_tfrecord_dir, f'{file_no}_data_{one_f_train_sample_num}.tfrecord')
                    test_tfrecord_file = os.path.join(test_tfrecord_dir, f'{file_no}_data_{one_f_test_sample_num}.tfrecord')

                    raw_dataset = tf.data.TFRecordDataset(abs_f)

                    writer_test = tf.data.experimental.TFRecordWriter(test_tfrecord_file)
                    writer_test.write(raw_dataset.take(one_f_test_sample_num))

                    writer_train = tf.data.experimental.TFRecordWriter(train_tfrecord_file)
                    writer_train.write(raw_dataset.skip(one_f_test_sample_num))

                    for _abs_f in all_tfrecord_file[file_no + 1:]:
                        shutil.copyfile(_abs_f, os.path.join(train_tfrecord_dir, os.path.basename(_abs_f)))
                    break

                file_no += 1

        elif strategy == 'random':
            sample_num_per_file = int(all_tfrecord_file[0].split('_')[-1].split('.')[0])

            train_file_no = 0
            test_file_no = 0
            train_sample_no = 0
            test_sample_no = 0

            train_sample = x_train[path_column].tolist()
            test_sample = x_test[path_column].tolist()

            raw_dataset = tf.data.TFRecordDataset(all_tfrecord_file)
            parsed_dataset = raw_dataset.map(_parse_image_function)

            try:
                for t1, t2 in parsed_dataset:
                    file_path = tf.compat.as_str_any(t1[path_column].numpy())
                    if file_path in train_sample:
                        if train_sample_no % sample_num_per_file == 0:
                            sample_num = sample_num_per_file if int(train_total_sample / sample_num_per_file) > train_file_no else train_total_sample % sample_num_per_file
                            train_tfrecord_file = os.path.join(train_tfrecord_dir, f'{train_file_no}_data_{sample_num}.tfrecord')
                            train_file_no += 1
                            writer_train = tf.io.TFRecordWriter(train_tfrecord_file)
                        writer_train.write(tf.compat.as_bytes(t2.numpy()))
                        train_sample_no += 1
                    if file_path in test_sample:
                        if test_sample_no % sample_num_per_file == 0:
                            sample_num = sample_num_per_file if int(test_total_sample / sample_num_per_file) > test_file_no else test_total_sample % sample_num_per_file
                            test_tfrecord_file = os.path.join(test_tfrecord_dir, f'{test_file_no}_data_{sample_num}.tfrecord')
                            test_file_no += 1
                            writer_test = tf.io.TFRecordWriter(test_tfrecord_file)
                        writer_test.write(tf.compat.as_bytes(t2.numpy()))
                        test_sample_no += 1
            except Exception as err:
                raise err
            finally:
                writer_train.close()
                writer_test.close()

    return 0

