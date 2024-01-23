import logging


class DataCanvas:

    def __init__(self):
        class Config:
            def __init__(self):
                class Params:
                    pass

                class Inputs:
                    pass

                class Outputs:
                    pass

                self.params = Params()
                self.inputs = Inputs()
                self.outputs = Outputs()

        self.conf = Config()
        self.logger = logging.getLogger("dc")

    def dataset(self, f_path):
        #     label_folder_type = ds.detail.labelFolderSettingType
        #     labelFolder = ds.detail.labelFolder
        #     dataFolder = ds.detail.dataFolder
        class Reader:
            def __init__(self):
                class Detail:
                    def __init__(self):
                        self.labelFolderSettingType = "FLODER_LABEL"
                        self.labelFolder="label"
                        self.dataFolder="image"
                self.detail = Detail()

            def read(self):
                import pandas as pd
                return pd.read_csv(f_path,sep='\t')

        return Reader()


dc = DataCanvas()
dc.conf.params.text_col = "text"
dc.conf.params.name_col = "attribute_name"
dc.conf.params.value_col = "attribute_value"
dc.conf.params.language = "chinese"
dc.conf.params.load_weight = "False"
dc.conf.params.num_train_epochs = 1
dc.conf.params.local_rank = -1
dc.conf.params.do_train = "True"
dc.conf.params.no_cuda = "False"
dc.conf.params.do_eval = "True"
dc.conf.params.learning_rate = 5e-5
dc.conf.params.gradient_accumulation_steps = 1
dc.conf.params.weight_decay = 0.01
dc.conf.params.adam_eps = 1e-6
dc.conf.params.adam_b1 = 1e-6
dc.conf.params.adam_b2 = 1e-6
dc.conf.params.adam_correct_bias = "True"
dc.conf.params.warmup_ratio = 0.06
dc.conf.params.lr_schedule = "warmup_linear"
dc.conf.params.batch_size = 2
dc.conf.params.max_grad_norm = 0.0
dc.conf.params.max_seq_length = 32
dc.conf.params.max_attr_length = 8
dc.conf.params.save_steps = 1
dc.conf.params.model_name = "best.pth"
dc.conf.params.label_list = ['B-a', 'I-a', 'O', '[CLS]', '[SEP]']


dc.conf.inputs.input_train = "./data/test_chinese.tsv"
dc.conf.inputs.input_test = "./data/test_chinese.tsv"
# dc.conf.inputs.input_train = "./data/zh_attribute_train.tsv"
# dc.conf.inputs.input_test = "./data/zh_attribute_test.tsv"
dc.conf.inputs.trained_model = ""
dc.conf.outputs.output_model = "./output"
dc.conf.outputs.prediction = "./output/prediction.csv"
dc.conf.outputs.logs = "./output/logs"
dc.conf.outputs.performance = "./output/performance.json"
