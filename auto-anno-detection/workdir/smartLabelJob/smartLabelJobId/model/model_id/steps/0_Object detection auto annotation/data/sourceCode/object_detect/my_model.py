from object_detect import Model


class MyModel(Model):

    @staticmethod
    def load_model(model_dir):
        print("===== Loading model...!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return MyModel()

    def __init__(self, name="MyModel"):
        self.name = name

    def adjust_model(self, *args, **kwargs):
        print("===== Adjusting model...")

    def save_model(self, save_dir):
        print("===== Saving model...")

    def train_model(self, df_train, df_val):
        print("===== Training model...")

    def predict(self, df_img):
        print("===== Predicting...")
        return df_img

