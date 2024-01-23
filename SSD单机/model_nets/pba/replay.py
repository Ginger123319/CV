from pathlib import Path
import json


class Schedule:
    def __init__(self, name, iteration, config, next_node=None):
        self.name = name
        self.iteration = iteration
        self.config = config
        self.next_node = next_node

    def __str__(self):
        explanation = "Config of schedule:"
        configs = self.export()
        for c in configs:
            explanation+="\n{}".format(c)
        return explanation

    def export(self, stretched_epochs=None):
        from collections import namedtuple
        Config = namedtuple("Config", field_names=["start", "end", "config"])
        retval = []
        pre = 0
        cur = self
        original_epochs = 0
        while cur:
            if cur.iteration==0:
                pre, cur = cur.iteration, cur.next_node
                continue
            retval.append(Config(pre, cur.iteration, cur.config))
            original_epochs = cur.iteration
            pre, cur = cur.iteration, cur.next_node
        if stretched_epochs is not None:
            print("Stretch epochs from {} to {}".format(original_epochs, stretched_epochs))
            ratio = stretched_epochs/original_epochs
            retval_new = []
            for r in retval:
                new_start, new_end = round(ratio*r.start), round(ratio*r.end)
                if new_start<new_end:
                    retval_new.append(Config(new_start, new_end, r.config))
                    print("Start end changes: {} {} -> {} {}".format(r.start, r.end, new_start, new_end))
                else:
                    print("Skip this config: {} {} -> {} {}".format(r.start, r.end, new_start, new_end))

            retval = retval_new
        return retval


def extract_schedule(log_dir, trial_name, original_epochs):
    # f = Path("ray_results/Trainable_2022-03-09_16-39-02/pbt_policy_5f345_00003.txt")
    print("Extract schedule from dir: {} Trial name: {}".format(log_dir, trial_name))
    pbt_policy_file = Path(log_dir, "pbt_policy_{}.txt".format(trial_name))
    if not pbt_policy_file.exists():
        result_file = Path("{}/{}/result.json".format(log_dir, trial_name))
        print("Policy file [{}] not exists, use [{}] instead.".format(pbt_policy_file, result_file))
        with open(result_file, "r") as f:
            config_json = json.loads(f.readline())
        schedule = Schedule(name=trial_name, iteration=original_epochs, config=config_json["config"]["aug_config"])
        return schedule

    policies_raw = pbt_policy_file.read_text(encoding="utf-8")
    policies = [json.loads(v) for v in policies_raw.split("\n") if v.strip() != ""]
    assert len(policies) > 0
    schedule = None
    pre = None
    for i in range(len(policies)):
        old_tag, new_tag, old_step, new_step, old_conf, new_conf = policies[i]
        cur_schedule = Schedule(new_tag, new_step, old_conf["aug_config"])
        if pre is not None:
            pre.next_node = cur_schedule
        if schedule is None:
            schedule = cur_schedule
        pre = cur_schedule
        if i == len(policies) - 1:
            cur_schedule.next_node = Schedule(old_tag, original_epochs, new_conf["aug_config"])
    print(schedule)
    return schedule


if __name__=="__main__":
    my_schedule = extract_schedule(log_dir="ray_results/Trainable_2022-03-10_16-56-44", trial_name="02a03_00000", original_epochs=30)
    print(my_schedule.export(50))
