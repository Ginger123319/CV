# -*- encoding: utf-8 -*-

import logging, sys

log_format = '%(asctime)s %(levelname)s [%(name)s] [%(thread)d-%(pathname)s:%(lineno)d] ===== %(message)s'
logging.basicConfig(level=logging.INFO,
                    format=log_format,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    # filename='/tmp/mr_log.log',
                    # filemode='a',
                    stream=sys.stdout)

    #shFormatter = logging.Formatter(fmt=log_format,
    #                                datefmt='%Y-%m-%d %H:%M:%S')
    #sh = logging.StreamHandler()
    # sh.setLevel(logging.INFO)
    # sh.setFormatter(shFormatter)
    # output log to console
    # self.logger.addHandler(sh)

logger = logging.getLogger("model_repo")
