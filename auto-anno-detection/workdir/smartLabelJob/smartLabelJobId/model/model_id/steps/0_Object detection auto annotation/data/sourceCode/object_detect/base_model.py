import abc


class Model:
    @abc.abstractmethod
    def adjust_model(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def save_model(self, save_dir):
        pass

    @staticmethod
    @abc.abstractmethod
    def load_model(model_dir):
        pass

    @abc.abstractmethod
    def train_model(self, df_train, df_val):
        pass

    @abc.abstractmethod
    def predict(self, df_img):
        pass
