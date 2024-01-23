import numpy as np
from PIL import Image
import random
from .aug_policies import apply_augment
from .aug_policies import OPS_NAMES
from pathlib import Path


def clip(value, floor, ceil):
    return max(min(value, ceil), floor)


def _load_image(image_path_or_array, width, height, aug_config=None):
    if isinstance(image_path_or_array, (str, Path)):
        img = Image.open(image_path_or_array)
    else:
        img = Image.fromarray(image_path_or_array)

    if isinstance(aug_config, list):
        # Do aug here
        cnt = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
        for _ in range(cnt):
            # aug_config is list of [AugName Prob Magnitude]
            aug_name, prob, magnitude = aug_config[random.randint(0, len(aug_config) - 1)]
            if random.random() < prob:
                img = apply_augment(img, aug_name, magnitude)
                # print("Apply augment: {}, {}".format(aug_name, magnitude))
    if isinstance(aug_config, dict):
        # Do randaug here
        for _ in range(aug_config["N"]):
            aug_name = OPS_NAMES[random.randint(0, len(OPS_NAMES) - 1)]
            magnitude = aug_config["M"]
            img = apply_augment(img, aug_name, magnitude)
    elif aug_config == "Random":
        cnt = random.choices([0, 1, 2], weights=[0.2, 0.3, 0.5])[0]
        for _ in range(cnt):
            # aug_config is list of [AugName Prob Magnitude]
            aug_name = OPS_NAMES[random.randint(0, len(aug_config) - 1)]
            magnitude = clip(random.gauss(0.5, 0.2), 0, 1)
            img = apply_augment(img, aug_name, magnitude)

    img = img.resize((width, height))
    image_arr = np.asarray(img, dtype=np.uint8)
    return image_arr


# def _load_images(image_paths, width, height):
#     return np.array([_load_image(image_path, width, height) for image_path in image_paths])


# def _img_gen(classes, img_paths, labels, width, height, aug_config):
#     class_dict = {c: i for i, c in enumerate(classes)}
#     print("CLASS DICT: {}".format(str(class_dict)))
#
#     def real_gen():
#         for img_path, img_label in zip(img_paths, labels):
#             img_arr = _load_image(img_path, width, height, aug_config)
#             lab = np.zeros(len(class_dict))
#             lab[class_dict[img_label]] = 1
#             yield img_arr, lab
#
#     return real_gen


def get_dataset(classes, record_path, image_col, label_col, width, height, aug_config, repeat, shuffle, batch_size,
                random_seed):
    import tensorflow as tf

    ds_tfrec = tf.data.TFRecordDataset(record_path)

    image_feature_description = {image_col: tf.io.FixedLenFeature([], tf.string),
                                 label_col: tf.io.FixedLenFeature([], tf.string),
                                 "shape": tf.io.FixedLenFeature([3], tf.int64)}

    def decode_example(x):
        example = tf.io.parse_single_example(x, image_feature_description)
        shape = example["shape"]
        # print("========type of shape: {} {} {}".format(shape.dtype, shape.shape, shape))
        img_buffer = example[image_col]
        # print("========type of img_buffer: {} {} {}".format(img_buffer.dtype, img_buffer.shape, type(img_buffer), type(img_buffer.numpy())))
        img_origin = np.frombuffer(img_buffer.numpy(), dtype=np.uint8)
        img_origin = img_origin.reshape(shape)
        img_aug = _load_image(img_origin, width=width, height=height, aug_config=aug_config)
        # print("========type of img_aug: {} {}".format(img_aug.dtype, img_aug.shape))
        img = tf.constant(img_aug, dtype=tf.int32)
        # print("========type of img: {} {}".format(type(img), img.shape))
        img = tf.clip_by_value(img, 0, 255)
        # print("========type of img clip: {} {}".format(type(img), img.shape))
        img = tf.cast(img, tf.uint8)
        # img.set_shape([height, width, 3])
        label = tf.cast((tf.constant(classes) == example[label_col]), tf.uint8)
        # label.set_shape(len(classes))
        return img, label

    # tf.TensorSpec(shape=[height, width, 3], dtype=tf.uint8)
    ds = ds_tfrec.map(lambda x: tf.py_function(func=decode_example, inp=[x], Tout=[tf.uint8, tf.uint8]),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _set_shape(img_tensor, label_tensor):
        img_tensor.set_shape([height, width, 3])
        label_tensor.set_shape([len(classes)])
        return img_tensor, label_tensor

    ds = ds.map(_set_shape, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if shuffle:
        # 不直接使用train_count 防止数据量过大. 设置成5个batch_size大小
        ds = ds.shuffle(buffer_size=batch_size * 5, seed=random_seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    if repeat:
        ds = ds.repeat()

    # tf.data.Options()` object then setting `options.experimental_distribute.auto_shard_policy = AutoShardPolicy.DATA` before applying the options object to the dataset via `dataset.with_options(options)`
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    ds = ds.with_options(options)

    return ds


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    import pandas as pd

    df_data = pd.read_csv("./dataset/data.csv")
    cur_classes = list(df_data["label"].unique())
    df_train, df_valid = train_test_split(df_data, shuffle=True, stratify=df_data["label"], test_size=0.2,
                                          random_state=1)

    config = {
        "train_df": df_train,
        "valid_df": df_valid,
        "path_col": "path",
        "label_col": "label",
        "classes": cur_classes,
        "width": 300,
        "height": 300,
        "aug_config": None,
        "batch_size": 16,
        "shuffle": True,
        "random_seed": 1,
        "model_type": "LeNet",  # VGG16 VGG19 ResNet50 Xception LeNet
        "use_pretrained_weights": True,
        "weights_dir": "xxx",
        "input_shape": (300, 300, 3),
        "class_num": 2,
        "optimizer_type": "SGD",  # SGD RMSprop Adagrad Adadelta Adam Adamax Nadam
        "learning_rate": 0.0001,
        "loss_fn_type": "categorical_crossentropy",
        "metrics": ["accuracy"]
    }
    ds_train = get_dataset(classes=config["classes"],
                           img_paths=config["train_df"][config["path_col"]],
                           labels=config["train_df"][config["label_col"]],
                           width=config["width"],
                           height=config["height"],
                           aug_config=config["aug_config"],
                           repeat=False,
                           shuffle=config["shuffle"],
                           batch_size=config["batch_size"],
                           random_seed=config["random_seed"])

    print(next(iter(ds_train)))
