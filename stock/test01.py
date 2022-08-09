import cfg

with open(cfg.data_dir) as f:
    _list = f.readlines()
    for i, elem in enumerate(_list):
        print(list(elem))
