import math
import numpy as np
import ray
import random
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from pathlib import Path
import torch
from torch.backends import cudnn
from torch import optim
from torch.utils.data import DataLoader
from model_nets.nets.ssd import get_ssd
from model_nets.nets.ssd_training import MultiBoxLoss
from model_nets.utils.dataloader import SSDDataset
from model_nets.utils.dataloader2 import SSDDataset2
from torch.autograd import Variable



def ssd_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        if box is None:
            continue
        images.append(img)
        bboxes.append(box)
    if len(bboxes)==0:
        return None, None
    images = np.array(images)
    return images, bboxes


def get_ds_from_config(config):
    if config["use_tfrecord"]:
        train_dataset = SSDDataset2(config["train_tf_record_file_pba"], config["train_tf_record_index_pba"], (config["min_dim"], config["min_dim"]), aug_config=config["aug_config"])
        ds_train = DataLoader(train_dataset, shuffle=True, batch_size=config["batch_size"], num_workers=4, pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)
        val_dataset = SSDDataset2(config["val_tf_record_file_pba"], config["val_tf_record_index_pba"], (config["min_dim"], config["min_dim"]))
        ds_valid = DataLoader(val_dataset, shuffle=False, batch_size=config["batch_size"], num_workers=4, pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)

    else:
        with open(config["annotation_path"]) as f:
            lines = f.readlines()
        num_val = int(len(lines) * config["val_size"])
        num_train = len(lines) - num_val

        train_dataset = SSDDataset(lines[:num_train], (config["min_dim"], config["min_dim"]), aug_config=config["aug_config"])
        ds_train = DataLoader(train_dataset, shuffle=False, batch_size=config["batch_size"], num_workers=4, pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)                
        val_dataset = SSDDataset(lines[num_train:], (config["min_dim"], config["min_dim"]))
        ds_valid = DataLoader(val_dataset, shuffle=False, batch_size=config["batch_size"], num_workers=4, pin_memory=True, drop_last=True, collate_fn=ssd_dataset_collate)        
    train_batch_cnt = max(1, len(train_dataset) // config["batch_size"])
    val_batch_cnt = max(1, len(val_dataset) // config["batch_size"])
    return ds_train, train_batch_cnt, ds_valid, val_batch_cnt

def get_model_from_config(config, weights=None):
    net = get_ssd("train", config["num_classes"])

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device of model: ", device)
    if weights is None:
        model_dict = net.state_dict()
        pretrained_dict = torch.load(config["model_path"], map_location=device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) ==  np.shape(v)}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)
    else:
        new_weights = {k.replace("module.", ""):v for k, v in weights.items()}
        net.load_state_dict(new_weights)
    print('Finished Loading weights!')

    for param in net.vgg.parameters():
        param.requires_grad = False

    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
        device_ids = []
        for i in range(gpu_num):
            device_ids.append(i)
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        cudnn.benchmark = True
        net = net.cuda()

    return net


def get_optimizer_from_config(config, net, weights=None):
    if config["optimizer"] == "SGD":
        optimizer = optim.SGD(net.parameters(), lr=config["lr"])
    else:
        optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    if weights:
        optimizer.load_state_dict(weights)
    return optimizer

def get_scheduler_from_config(config, optimizer, weights=None):
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.92)
    if weights:
        lr_scheduler.load_state_dict(weights)
    return lr_scheduler

def train_model(model, optimizer, scheduler, config, initial_epoch, epochs):
    if model is None:
        model = get_model_from_config(config=config)
        optimizer = get_optimizer_from_config(config=config, net=model)
        scheduler = get_scheduler_from_config(config=config, optimizer=optimizer)

    ds_train, train_batch_cnt, ds_valid, valid_batch_cnt = get_ds_from_config(config=config)
    criterion = MultiBoxLoss(config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, torch.cuda.is_available())

    loc_loss = 0
    conf_loss = 0
    loc_loss_val = 0
    conf_loss_val = 0

    model.train()
    print('Start Train')

    cuda = torch.cuda.is_available()
    for epoch_id in range(initial_epoch, epochs):
        for batch_id, batch in enumerate(ds_train):
            images, targets = batch[0], batch[1]
            if targets is None:
                print(f"Skip batch {iteration}: no label")
                continue
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            out = model(images)
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()            
            print("Training... Epoch {}/{} Batch {}/{}: loss_l {}  ---  loss_c {} --- loss {}"
                  .format(epoch_id+1, epochs, batch_id+1, train_batch_cnt, loss_l.item(), loss_c.item(), loss.item()))
        scheduler.step()
        print("Train loss (average of all batches): loss_l {} --- loss_c {}".format(loc_loss/train_batch_cnt, conf_loss/train_batch_cnt))

        model.eval()
        print("Start Validation:")
        for batch_id, batch in enumerate(ds_valid):
            images, targets = batch[0], batch[1]
            if targets is None:
                print(f"Skip batch {iteration}: no label")
                continue
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                out = model(images)
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loc_loss_val += loss_l.item()
                conf_loss_val += loss_c.item()
                print("Validating... Epoch {}/{} Batch {}/{}: loss_l {}  ---  loss_c {} --- loss {}"
                  .format(epoch_id+1, epochs, batch_id+1, valid_batch_cnt, loss_l.item(), loss_c.item(), loss.item()))
        print("Validation loss (average of all batches): loss_l {} --- loss_c {}".format(loc_loss_val/valid_batch_cnt, conf_loss_val/valid_batch_cnt))
    return (loc_loss_val+conf_loss_val)/valid_batch_cnt


class Trainable(tune.Trainable):

    def status(self):
        content = ["id: {}".format(id(self)), "name: {}".format(self.trial_name), "iter: {}".format(self.iteration)]

        def get_c(name):
            return self.config.get(name)

        content.extend(["{}: {}".format(v, get_c(v)) for v in ["aug_config"]])
        return "[{}]".format(" | ".join(content))

    def setup(self, config):
        print("Invoke setup:{}".format(self.status()))
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self._initial_epoch = 0

    def step(self):  # This is called iteratively.
        print("Invoke step: {}".format(self.status()))
        print("Epoch: {} + {}".format(self._initial_epoch, 1))
        if self.model is None:
            self.model = get_model_from_config(config=self.config)
            self.optimizer = get_optimizer_from_config(config=self.config, net=self.model)
            self.scheduler = get_scheduler_from_config(config=self.config, optimizer=self.optimizer)
        val_loss = train_model(model=self.model, optimizer=self.optimizer, scheduler=self.scheduler, config=self.config, initial_epoch=self._initial_epoch, epochs=self._initial_epoch+1)
        self._initial_epoch+=1
        return {"score": val_loss}


    def save_checkpoint(self, tmp_checkpoint_dir):
        print("Invoke save_checkpoint:{} : {}".format(tmp_checkpoint_dir, self.status()))
        save_path = str(Path(tmp_checkpoint_dir, "simple.h5"))
        weights = {"model": self.model.state_dict(),
                   "optimizer": self.optimizer.state_dict(),
                   "scheduler": self.scheduler.state_dict()}
        torch.save(weights, save_path)
        Path(tmp_checkpoint_dir, "history").write_text(str(self._initial_epoch), encoding="utf-8")
        return tmp_checkpoint_dir

    def load_checkpoint(self, checkpoint):
        print("Invoke load_checkpoint:{} : {}".format(checkpoint, self.status()))
        from tensorflow.keras.models import load_model
        
        weights = torch.load(str(Path(checkpoint, "simple.h5")))
        # TODO get model optimizer scheduler from weigths
        self.model = get_model_from_config(config=self.config, weights=weights["model"])
        self.optimizer = get_optimizer_from_config(config=self.config, net=self.model, weights=weights["optimizer"])
        self.scheduler = get_scheduler_from_config(config=self.config, optimizer=self.optimizer, weights=weights["scheduler"])
        self._initial_epoch = int(Path(checkpoint, "history").read_text(encoding="utf-8"))
        print("Invoke load_checkpoint end:{} ".format(self.status()))


def explore_fn(config):
    print("Invoke explore_fn: {}".format(config["aug_config"]))

    def clip(value, floor, ceil):
        return max(min(value, ceil), floor)

    for _i, _c in enumerate(config["aug_config"]):
        cur_policy = config["aug_config"][_i]
        if random.random() < 0.2:
            # cur_policy[1] = random.random()
            # cur_policy[2] = random.random()
            cur_policy[1] = random.choice([v/10. for v in range(0,11)])
            cur_policy[2] = random.choice([v/10. for v in range(0,11)])
        else:
            prob_add = random.choice([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
            # prob_add = random.gauss(0, 0.15)
            cur_policy[1] = clip(cur_policy[1] + prob_add, 0, 1)
            magnitude_add = random.choice([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
            # magnitude_add = random.gauss(0, 0.15)
            cur_policy[2] = clip(cur_policy[2] + magnitude_add, 0, 1)
    print("Invoke explore_fn result: {}".format(config["aug_config"]))
    return config


def pba_train(train_config, export_epochs):

    ray.init(local_mode=False, num_cpus=16, num_gpus=2, ignore_reinit_error=True)
    # ray.init(local_mode=True, num_cpus=8, num_gpus=1, ignore_reinit_error=True)

    # import logging
    # logging.Logger.manager.loggerDict["ray"].setLevel("DEBUG")
    print("Train_config of pba:\n", train_config)
    
    from .aug_policies import generate_policies
    train_config["aug_config"] = generate_policies()  # ray.tune.sample_from(generate_policies)

    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="score",
        mode="min",
        perturbation_interval=1,  # every ? `time_attr` units
        custom_explore_fn=explore_fn,
        log_config=True,
        quantile_fraction=0.25,
        synch=True)  # synch=True避免多一次持久化

    def trial_name_fn(trial):
        return "{}".format(trial.trial_id)

    def trial_dirname_fn(trial):
        return "{}".format(trial.trial_id)

    analysis = tune.run(
        Trainable,
        name="pba",
        trial_name_creator=trial_name_fn,
        trial_dirname_creator=trial_dirname_fn,
        config=train_config,
        scheduler=scheduler,
        stop={"training_iteration": train_config["epochs"]},
        resources_per_trial={"cpu": 8, "gpu": 1},
        num_samples=train_config["population_size"],
        reuse_actors=False,
        local_dir=train_config["local_dir"],
        # metric="score",
        # mode="max",
        # keep_checkpoints_num=10,
        # checkpoint_score_attr="score",
        verbose=3,
    )

    print("Replay the best trial...")
    from .replay import extract_schedule
    log_dir = analysis._experiment_states[0]["runner_data"]["_local_checkpoint_dir"]
    best_trial_name = analysis.get_best_trial(metric="score", mode="max")
    best_schedule = extract_schedule(log_dir=log_dir, trial_name=best_trial_name, original_epochs=train_config["epochs"]).export(export_epochs)
    return best_schedule
