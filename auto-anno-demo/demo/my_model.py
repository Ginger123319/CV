from pathlib import Path
from main_utils import Model
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    print("Append sys path: {}".format(ROOT.resolve()))
    sys.path.append(str(ROOT))  # add ROOT to PATH


class MyModel(Model):

    @staticmethod
    def load_model(model_dir):
        print("===== Loading model...")
        m = MyModel()
        m.weights = str(Path(model_dir, "weights.pt").resolve())
        return m

    def __init__(self, name="MyModel"):
        self.name = name
        self.weights = "TODO"

    def save_model(self, save_dir):
        print("Saving model to {}".format(save_dir))

    def train_model(self, df_train, df_val, work_dir, is_first_train, **options):
        print("Training...")

    def query_hard_example(self, df_img, work_dir, query_cnt=100, strategy="LeastConfidence", **options):
        print("===== Predicting...")
        df_pred = df_img
        df_pred["lable"] = ["annotation_{}".format(i) for i in range(len(df_img))]
        df_pred["isHard"] = 0
        df_pred.loc[:query_cnt, "isHard"] = 1
        return df_pred
