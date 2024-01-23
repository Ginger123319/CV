import os
import math
import ray
import random
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from pathlib import Path



def get_ds_from_config(config):
    from .dataset_utils import get_dataset

    ds_train = get_dataset(classes=config["classes"],
                           record_path=config["ds_train_path"],
                           image_col=config["image_col"],
                           label_col=config["label_col"],
                           width=config["width"],
                           height=config["height"],
                           aug_config=config["aug_config"],
                           repeat=True,
                           shuffle=config["shuffle"],
                           batch_size=config["batch_size"],
                           random_seed=config["random_seed"])
    ds_valid = get_dataset(classes=config["classes"],
                           record_path=config["ds_valid_path"],
                           image_col=config["image_col"],
                           label_col=config["label_col"],
                           width=config["width"],
                           height=config["height"],
                           aug_config=None,
                           repeat=True,
                           shuffle=False,
                           batch_size=config["batch_size"],
                           random_seed=config["random_seed"])
    return ds_train, ds_valid


def train_model(model, config, initial_epoch, epochs, callbacks=None):
    if model is None:
        from .model_utils import get_compiled_model
        model = get_compiled_model(model_type=config["model_type"],
                                   use_pretrained_weights=config["use_pretrained_weights"],
                                   weights_dir=config["weights_dir"],
                                   input_shape=config["input_shape"],
                                   class_num=config["class_num"],
                                   optimizer_type=config["optimizer_type"],
                                   learning_rate=config["learning_rate"],
                                   loss_fn_type=config["loss_fn_type"],
                                   metrics=config["metrics"],
                                   activation_function=config.get("activation_function", "relu"))

    ds_train, ds_valid = get_ds_from_config(config)

    history = model.fit(ds_train,
                        initial_epoch=initial_epoch,
                        epochs=epochs,
                        verbose=1,
                        # batch_size=config["batch_size"],
                        steps_per_epoch=config["batch_count_per_epoch"],
                        callbacks=callbacks)

    metric_name = model.metrics_names[0]
    train_loss, train_metric = history.history["loss"][0], history.history[metric_name][0]
    loss_and_metrics = model.evaluate(ds_valid, verbose=1, steps=config["valid_batch_count_per_epoch"])
    val_loss, val_metric = loss_and_metrics
    print(f"Epoch {initial_epoch} - loss: {train_loss} - {metric_name}: {train_metric} - val_loss: {val_loss} - val_{metric_name}: {val_metric}")
    print(model.metrics_names, loss_and_metrics)
    return model, loss_and_metrics


class Trainable(tune.Trainable):

    def status(self):
        content = ["id: {}".format(id(self)), "name: {}".format(self.trial_name), "iter: {}".format(self.iteration)]

        def get_c(name):
            return self.config.get(name)

        content.extend(["{}: {}".format(v, get_c(v)) for v in ["aug_config"]])
        return "[{}]".format(" | ".join(content))

    def setup(self, config):
        print("Invoke setup:{}".format(self.status()))
        self.model = None
        self._initial_epoch = 0

        from .model_utils import set_gpu_memory
        set_gpu_memory()

    def step(self):  # This is called iteratively.
        print("Invoke step: {}".format(self.status()))
        print("Epoch: {} + {}".format(self._initial_epoch, self.config["epochs_per_step"]))

        self.model, loss_or_metrics = train_model(self.model, self.config, initial_epoch=self._initial_epoch, epochs=self._initial_epoch + self.config["epochs_per_step"])
        self._initial_epoch += self.config["epochs_per_step"]

        return {"score": loss_or_metrics[1]}

    def save_checkpoint(self, tmp_checkpoint_dir):
        print("Invoke save_checkpoint:{} : {}".format(tmp_checkpoint_dir, self.status()))
        save_path = str(Path(tmp_checkpoint_dir, "simple.h5"))
        self.model.save(save_path)
        Path(tmp_checkpoint_dir, "history").write_text(str(self._initial_epoch), encoding="utf-8")
        return tmp_checkpoint_dir

    def load_checkpoint(self, checkpoint):
        print("Invoke load_checkpoint:{} : {}".format(checkpoint, self.status()))
        from tensorflow.keras.models import load_model
        self.model = load_model(str(Path(checkpoint, "simple.h5")))
        self._initial_epoch = int(Path(checkpoint, "history").read_text(encoding="utf-8"))
        print("Invoke load_checkpoint end:{} ".format(self.status()))


def explore_fn(config):
    print("Invoke explore_fn: {}".format(config["aug_config"]))

    def clip(value, floor, ceil):
        return max(min(value, ceil), floor)

    for _i, _c in enumerate(config["aug_config"]):
        cur_policy = config["aug_config"][_i]
        if random.random() < 0.2:
            cur_policy[1] = random.random()
            cur_policy[2] = random.random()
        else:
            prob_add = random.choice([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
            #prob_add = random.gauss(0, 0.15)
            cur_policy[1] = clip(cur_policy[1] + prob_add, 0, 1)
            magnitude_add = random.choice([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
            #magnitude_add = random.gauss(0, 0.15)
            cur_policy[2] = clip(cur_policy[2] + magnitude_add, 0, 1)
    print("Invoke explore_fn result: {}".format(config["aug_config"]))
    return config


def pba_train(train_config, export_epochs):
    if train_config["auto_aug_config"] is not None:
        from .replay import load_schedule
        return load_schedule(train_config["auto_aug_config"], stretched_epochs=export_epochs)
    ray.shutdown()
    import tensorflow as tf
    physical_devices = tf.config.list_physical_devices('GPU')
    print("GPU devices:", physical_devices)
    num_gpus = max(len(physical_devices),1)
    ray.init(local_mode=False, num_gpus=num_gpus)
    from .aug_policies import generate_policies
    train_config["aug_config"] = generate_policies()  # ray.tune.sample_from(generate_policies)

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="score",
        mode="max",
        perturbation_interval=1,  # every ? `time_attr` units
        custom_explore_fn=explore_fn,
        log_config=True,
        quantile_fraction=0.4,
        synch=True)  # synch=True避免多一次持久化

    def trial_name_fn(trial):
        return "{}".format(trial.trial_id)

    def trial_dirname_fn(trial):
        return "{}".format(trial.trial_id)

    analysis = tune.run(
        Trainable,
        # name="pba",
        trial_name_creator=trial_name_fn,
        trial_dirname_creator=trial_dirname_fn,
        config=train_config,
        scheduler=scheduler,
        stop={"training_iteration": math.ceil(train_config["epochs"] / train_config["epochs_per_step"])},
        resources_per_trial={"gpu": 1},
        num_samples=train_config["population_size"],
        reuse_actors=False,
        local_dir=train_config["local_dir"],
        # metric="score",
        # mode="max",
        # keep_checkpoints_num=10,
        # checkpoint_score_attr="score",
        verbose=3,
        max_failures=0,
        fail_fast=True,
        log_to_file=True
    )

    print("Replay the best trial...")
    from .replay import extract_schedule, save_schedule
    log_dir = analysis._experiment_states[0]["runner_data"]["_local_checkpoint_dir"]
    best_trial_name = analysis.get_best_trial(metric="score", mode="max")
    best_schedule = extract_schedule(log_dir=log_dir, trial_name=best_trial_name, original_epochs=train_config["epochs"]).export(export_epochs)
    save_schedule(best_schedule, Path(train_config["local_dir"], "best_schedule.txt"))
    return best_schedule


def train_model_with_schedule(config, schedule):
    from .model_utils import set_gpu_memory
    set_gpu_memory()

    import tensorflow as tf
    if config["use_amp"]:
        print("启用混合精度训练")
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy('mixed_float16')
    update_freq = max(config["batch_count_per_epoch"] // 10, 1)
    print("Tensorboard update_freq={}".format(update_freq))

    tensorboard_log = tf.keras.callbacks.TensorBoard(log_dir=config["tensorboard_dir"],
                                                     histogram_freq=1,
                                                     write_graph=False,
                                                     write_images=False,
                                                     update_freq=update_freq,
                                                     profile_batch=0,
                                                     embeddings_freq=0)
    if isinstance(schedule, list):
        writer = tf.summary.create_file_writer(str(Path(config["tensorboard_dir"], "validation").absolute()))
        retrained_model = None
        losses = []
        with writer.as_default():
            for s in schedule:
                for ss in range(s.start, s.end):
                    config["aug_config"] = s.config
                    retrained_model, loss_and_metrics = train_model(model=retrained_model, config=config, initial_epoch=ss, epochs=ss+1)
                    losses.append(loss_and_metrics[0])
                    check_periods = config["early_stop"]
                    tf.summary.scalar("epoch_loss", loss_and_metrics[0], step=ss+1)
                    tf.summary.scalar("epoch_accuracy", loss_and_metrics[1], step=ss+1)
                    if config["early_stop"]>0 and len(losses) > check_periods:
                        stop = True
                        for i in range(1, check_periods + 1):
                            if losses[-i] <= losses[-i - 1]:
                                stop = False
                                break
                        if stop:
                            print("Loss has been increased for {} rounds, stop now.".format(check_periods))

        model = retrained_model
        history = losses
    elif schedule == "RandAug":
        import uuid
        from collections import namedtuple
        
        from datacanvas.aps import dc
        from dc_model_repo import model_repo_client
        work_dir = model_repo_client.get_dc_proxy().get_work_dir()
        work_dir = os.path.join(work_dir, dc.conf.global_params.block_id, "work_files")
        
        model_dir = work_dir + "/model_img_cls/"+str(uuid.uuid4())
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        n_m = []
        for p_n in [1, 2]:
            for p_m in [v/10 for v in range(0,11)]:
                n_m.append((p_n, p_m))

        import multiprocessing
        from multiprocessing import Array
        import ctypes   


        def bytes2obj(bs):
            import json
            str_len = int.from_bytes(bs[:2], byteorder="big", signed=False)
            return json.loads(bs[2:2 + str_len].decode(encoding="utf-8"))


        def obj2bytes(s, bytes_length):
            assert bytes_length <= 65537
            import json
            bs = bytearray(bytes_length)
            content = json.dumps(s).encode(encoding="utf-8")
            if len(content) > bytes_length - 2:
                raise ValueError("bytes_length太小了")
            bs[0:2] = int.to_bytes(len(content), length=2, byteorder="big", signed=False)
            bs[2:2 + len(content)] = content
            return bytes(bs)

        def train_one_trail(results, i, p_n, p_m):
            print("Train one trail:", i, p_n, p_m)
            cur_name = "N_{}_M_{}".format(p_n, p_m)
            config["aug_config"] = {"N": p_n, "M": p_m}
            callbacks = [tf.keras.callbacks.TensorBoard(log_dir=Path(config["tensorboard_dir"],cur_name),
                                                        histogram_freq=1,
                                                        write_graph=False,
                                                        write_images=False,
                                                        update_freq=update_freq,
                                                        profile_batch=0,
                                                        embeddings_freq=0)]
            if config["early_stop"] > 0:
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', min_delta=0.00001, patience=5, verbose=0, mode='auto',
                    baseline=None, restore_best_weights=True
                )
                callbacks.append(early_stop)
            print("callbacks done.")
            
            
            from .model_utils import get_compiled_model
            print("Start compile...")
            try:
                model = get_compiled_model(model_type=config["model_type"],
                                            use_pretrained_weights=config["use_pretrained_weights"],
                                            weights_dir=config["weights_dir"],
                                            input_shape=config["input_shape"],
                                            class_num=config["class_num"],
                                            optimizer_type=config["optimizer_type"],
                                            learning_rate=config["learning_rate"],
                                            loss_fn_type=config["loss_fn_type"],
                                            metrics=config["metrics"],
                                            activation_function=config.get("activation_function", "relu"))
            except Exception as e:
                print("error", e)
                raise e
            print("model compile done.")
            
            ds_train, ds_valid = get_ds_from_config(config)

            history = model.fit(ds_train,
                                validation_data=ds_valid,
                                initial_epoch=0,
                                epochs=config["epochs"],
                                verbose=1,
                                steps_per_epoch=config["batch_count_per_epoch"],
                                validation_steps=config["valid_batch_count_per_epoch"],
                                callbacks=callbacks)

            metric_name = model.metrics_names[1]
            train_loss, train_metric = history.history["loss"][-1], history.history[metric_name][-1]
            val_loss, val_metric = model.evaluate(ds_valid, verbose=1, steps=config["valid_batch_count_per_epoch"])
            model_path = str(Path(model_dir, f"{cur_name}_loss_{val_loss}_{metric_name}_{val_metric}.h5"))
            model.save(model_path)
            tmp = bytes2obj(bytes(results))
            tmp.append([cur_name, train_loss, train_metric, val_loss, val_metric, model_path])
            results[:] = obj2bytes(tmp, 50000)
            del model, history
            print(f"RandAug {i+1}/{len(n_m)} {cur_name} - loss: {train_loss} - {metric_name}: {train_metric} - val_loss: {val_loss} - val_{metric_name}: {val_metric}. Saving to: {model_path}")

        train_results = Array(ctypes.c_ubyte, obj2bytes([], 50000))

        for i, (p_n, p_m) in enumerate(n_m):
            if len(tf.config.list_physical_devices('GPU'))>0:
                train_one_trail(train_results,i, p_n, p_m)
            else:
                process_eval = multiprocessing.Process(target=train_one_trail, args=(train_results,i, p_n, p_m))
                process_eval.start()
                process_eval.join()

        train_info = bytes2obj(bytes(train_results))
        results_sorted = sorted(train_info, key=lambda v: v[4], reverse=True)
        best_one = results_sorted[0]
        print("The best trail: {}".format(best_one))
        return tf.keras.models.load_model(filepath=best_one[5]), None
    else:
        assert schedule in [None, "Random"]
        callbacks = [tensorboard_log]
        if config["early_stop"]>0:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', min_delta=0.00001, patience=5, verbose=0, mode='auto',
                baseline=None, restore_best_weights=True
            )
            callbacks.append(early_stop)
        config["aug_config"] = schedule
        from .model_utils import get_compiled_model
        model = get_compiled_model(model_type=config["model_type"],
                                   use_pretrained_weights=config["use_pretrained_weights"],
                                   weights_dir=config["weights_dir"],
                                   input_shape=config["input_shape"],
                                   class_num=config["class_num"],
                                   optimizer_type=config["optimizer_type"],
                                   learning_rate=config["learning_rate"],
                                   loss_fn_type=config["loss_fn_type"],
                                   metrics=config["metrics"],
                                   activation_function=config.get("activation_function", "relu"))

        ds_train, ds_valid = get_ds_from_config(config)

        # for d1, d2 in ds_train.take(2):
        #     print("======", d1.shape, d1.dtype, d2.shape, d2.dtype)
        #     print("======", d2)
        # raise Exception("Stop!")

        history = model.fit(ds_train,
                            validation_data=ds_valid,
                            initial_epoch=0,
                            epochs=config["epochs"],
                            verbose=1,
                            # batch_size=config["batch_size"],
                            steps_per_epoch=config["batch_count_per_epoch"],
                            validation_steps=config["valid_batch_count_per_epoch"],
                            callbacks=callbacks)

    return model, history
