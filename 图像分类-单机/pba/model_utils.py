def set_gpu_memory():
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("Set memory growth ok: {} Physical GPUs. {} Logical GPUs".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print("Set memory growth failed: {} Physical GPUs.".format(len(gpus)))
            print(e)


class ModelConfig:
    def __init__(self, model_type, use_pretrained_weights, weights_dir, input_shape, class_num):
        self.model_type = model_type
        self.use_pretrained_weights = use_pretrained_weights
        self.weights_dir = weights_dir
        self.input_shape = input_shape
        self.class_num = class_num


def get_model(model_type, use_pretrained_weights, weights_dir, input_shape, class_num, activation_function="relu"):
    import os
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model_config = ModelConfig(model_type, use_pretrained_weights, weights_dir, input_shape, class_num)
    print("要构造的模型：{}".format(model_config.model_type))
    # print("Weights dir: {}".format(weights_dir), os.path.join(model_config.weights_dir, "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"))

    # print("======= Input shape: ", model_config.input_shape)
    # from pathlib import Path
    # for p in Path(model_config.weights_dir).iterdir():
    #    print("Weigths file:", p)

    if model_config.model_type == "VGG16":
        if model_config.use_pretrained_weights:
            base_model = tf.keras.applications.VGG16(
                include_top=False, weights=os.path.join(model_config.weights_dir, "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"),
                input_tensor=None, input_shape=model_config.input_shape, pooling=None, classes=model_config.class_num
            )
            base_model.trainable = False
        else:
            base_model = tf.keras.applications.VGG16(
                include_top=False, weights=None,
                input_tensor=None, input_shape=model_config.input_shape, pooling=None, classes=model_config.class_num
            )
            base_model.trainable = True
        inputs = keras.Input(shape=model_config.input_shape)
        x = tf.keras.applications.vgg16.preprocess_input(inputs)
        if model_config.use_pretrained_weights:
            x = base_model(x, training=False)
        else:
            x = base_model(x)
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation=activation_function, name='fc1')(x)
        x = layers.Dense(4096, activation=activation_function, name='fc2')(x)
        outputs = layers.Dense(model_config.class_num, activation='softmax', name='predictions')(x)
        return keras.Model(inputs, outputs)
    elif model_config.model_type == "VGG19":
        if model_config.use_pretrained_weights:
            base_model = tf.keras.applications.VGG19(
                include_top=False, weights=os.path.join(model_config.weights_dir, "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5"),
                input_tensor=None, input_shape=model_config.input_shape, pooling=None, classes=model_config.class_num)
            base_model.trainable = False
        else:
            base_model = tf.keras.applications.VGG19(
                include_top=False, weights=None,
                input_tensor=None, input_shape=model_config.input_shape, pooling=None, classes=model_config.class_num)
            base_model.trainable = True
        inputs = keras.Input(shape=model_config.input_shape)
        x = tf.keras.applications.vgg19.preprocess_input(inputs)
        if model_config.use_pretrained_weights:
            x = base_model(x, training=False)
        else:
            x = base_model(x)
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation=activation_function, name='fc1')(x)
        x = layers.Dense(4096, activation=activation_function, name='fc2')(x)
        outputs = layers.Dense(model_config.class_num, activation='softmax', name='predictions')(x)
        return keras.Model(inputs, outputs)
    elif model_config.model_type == "ResNet50":
        if model_config.use_pretrained_weights:
            base_model = tf.keras.applications.ResNet50(
                include_top=False, weights=os.path.join(model_config.weights_dir, "resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"),
                input_tensor=None, input_shape=model_config.input_shape, pooling=None, classes=model_config.class_num)
            base_model.trainable = False
        else:
            base_model = tf.keras.applications.ResNet50(
                include_top=False, weights=None,
                input_tensor=None, input_shape=model_config.input_shape, pooling=None, classes=model_config.class_num)
            base_model.trainable = True
        inputs = keras.Input(shape=model_config.input_shape)
        x = tf.keras.applications.resnet50.preprocess_input(inputs)
        if model_config.use_pretrained_weights:
            x = base_model(x, training=False)
        else:
            x = base_model(x)
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        outputs = layers.Dense(model_config.class_num, activation='softmax', name='probs')(x)
        return keras.Model(inputs, outputs)
    elif model_config.model_type == "Xception":
        if model_config.use_pretrained_weights:
            base_model = tf.keras.applications.Xception(
                include_top=False, weights=os.path.join(model_config.weights_dir, "xception_weights_tf_dim_ordering_tf_kernels_notop.h5"),
                input_tensor=None, input_shape=model_config.input_shape, pooling=None, classes=model_config.class_num)
            base_model.trainable = False
        else:
            base_model = tf.keras.applications.Xception(
                include_top=False, weights=None,
                input_tensor=None, input_shape=model_config.input_shape, pooling=None, classes=model_config.class_num)
            base_model.trainable = True
        inputs = keras.Input(shape=model_config.input_shape)
        x = tf.keras.applications.xception.preprocess_input(inputs)
        if model_config.use_pretrained_weights:
            x = base_model(x, training=False)
        else:
            x = base_model(x)
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        outputs = layers.Dense(model_config.class_num, activation='softmax', name='predictions')(x)
        return keras.Model(inputs, outputs)
    elif model_config.model_type == "LeNet":
        if model_config.use_pretrained_weights:
            print("当前LeNet模型不支持加载预训练权重")
        model = tf.keras.Sequential()
        model.add(keras.layers.Input(shape=model_config.input_shape))
        model.add(keras.layers.Conv2D(filters=6, kernel_size=(5, 5), strides=(1, 1), padding="valid"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
        model.add(keras.layers.Activation(activation_function))

        model.add(keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding="valid"))
        model.add(keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding="valid"))
        model.add(keras.layers.Activation(activation_function))

        model.add(keras.layers.Conv2D(filters=120, kernel_size=(5, 5), strides=(1, 1), padding="valid"))
        model.add(keras.layers.Activation(activation_function))

        model.add(keras.layers.Flatten())

        model.add(keras.layers.Dense(units=84))
        model.add(keras.layers.Activation(activation_function))

        model.add(keras.layers.Dense(units=model_config.class_num))
        model.add(keras.layers.Activation("softmax"))
        return model
    else:
        raise Exception("不支持这种模型：{}".format(model_config.model_type))


def get_loss_fn(loss_fn_type):
    from tensorflow import keras
    if loss_fn_type == "categorical_crossentropy":
        return keras.losses.CategoricalCrossentropy(from_logits=False)
    else:
        raise Exception("不支持这种损失函数：{}".format(loss_fn_type))


def get_optimizer(optimizer_type, learning_rate):
    import tensorflow as tf
    # logger.info("要使用的优化器：{}".format(optimizer_type))
    if optimizer_type == "SGD":
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.2)
    elif optimizer_type == "RMSprop":
        return tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
    elif optimizer_type == "Adagrad":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_type == "Adadelta":
        return tf.keras.optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_type == "Adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_type == "Adamax":
        return tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    elif optimizer_type == "Nadam":
        return tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    else:
        raise Exception("不支持这种优化器：{}".format(optimizer_type))


def get_compiled_model(model_type, use_pretrained_weights, weights_dir, input_shape, class_num,
                       optimizer_type, learning_rate, loss_fn_type, metrics, activation_function="relu"):
    import tensorflow as tf
    optimizer = get_optimizer(optimizer_type, learning_rate)
    loss_fn = get_loss_fn(loss_fn_type)
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = get_model(model_type, use_pretrained_weights, weights_dir, input_shape, class_num, activation_function=activation_function)
        model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
    return model


if __name__=="__main__":
    model_str_list = ["LeNet",     #  3,317,166
                      "VGG16",     # 50,382,658
                      "VGG19",     # 55,692,354
                      "ResNet50",  # 23,591,810
                      "Xception"]  # 20,865,578

    for m in model_str_list:
        print("================================================\n{}".format(m))
        mm = get_compiled_model(m, False, None, [100,100,3], 2, "Adam", 0.001, "categorical_crossentropy", "accuracy")
        mm.summary()
