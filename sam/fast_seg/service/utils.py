import logging, sys
from datetime import datetime
from pathlib import Path

formatter = logging.Formatter(fmt='(%(levelname)s) %(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')    

log = logging.getLogger("app")
log_dir = "logs"
Path(log_dir).mkdir(parents=True, exist_ok=True)
log_file = "app_{}.log".format(datetime.strftime(datetime.today(), '%Y-%m-%d_%H-%M-%S'))
fh = logging.FileHandler(filename=Path(log_dir, log_file), encoding="utf-8")
fh.setFormatter(formatter)
log.addHandler(fh)
sh = logging.StreamHandler(stream=sys.stdout)
sh.setFormatter(formatter)
log.addHandler(sh)
log.setLevel(logging.DEBUG)

import time

class Timer:
    def _time(self):
        return f"{int((time.time()-self.start)*1000)}ms"

    def __init__(self, name=""):
        self.name = name
        self.start = time.time()
        log.info(f"=== {self.name} 开始计时……")
    def __str__(self):
        return f"=== {self.name} 耗时 {self._time()}"
    def __call__(self, msg=""):
        return f"=== {self.name} --> {msg} 耗时 {self._time()}"
