from datetime import datetime
import asyncio
import json
from pathlib import Path
from tornado.web import Application, RequestHandler, url
import cv2
import numpy as np
from service.utils import Timer, log
import torch
import uuid

uuid.uuid4

time_str = datetime.strftime(datetime.today(), '%Y-%m-%d %H:%M:%S')
from cachetools import LRUCache
# 定义一个缓存对象，最大容量为100
global cache 
# 可用于缓存使用的内存大小（MB）
memory_available = 1000
max_size = memory_available // 52
cache = LRUCache(maxsize=max_size)


class MainHandler(RequestHandler):
    def get(self):
        self.write(
            f"FAST SEG. Start time: {time_str} Device: {DEVICE} Now: {datetime.strftime(datetime.today(), '%Y-%m-%d %H:%M:%S')}")


def check_image(img, channel_cnt=3):
    newly_read = isinstance(img, str)

    if newly_read:
        assert Path(img).exists(), f"Image not exists: {img}"
        t = Timer(f"Read img {img}")
        img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        log.info(t)
    shape = img.shape
    if len(shape) != 3:
        raise ValueError("图片shape异常")
    if shape[0]*shape[1]*shape[2] == 0:
        raise ValueError("图片大小异常")
    if shape[2] != channel_cnt:
        raise ValueError(f"只支持{channel_cnt}通道图片")
    if newly_read:
        if channel_cnt == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            log.info(t("颜色转换"))
    return img


def get_bndbox(arr):
    m = arr[:, :, :3].max(axis=-1)
    h = m.sum(axis=1) > 0
    w = m.sum(axis=0) > 0
    assert h.sum() > 0

    x_min = int(np.argmax(w))
    y_min = int(np.argmax(h))
    x_max = int(w.shape[0]-np.argmax(w[::-1])-1)
    y_max = int(h.shape[0]-np.argmax(h[::-1])-1)
    if x_min >= x_max or y_min >= y_max:
        print(
            f'Error bndbox: {"xmin":x_min, "ymin":y_min, "xmax":x_max, "ymax":y_max}')
        raise Exception(
            f'Error bndbox: {"xmin":x_min, "ymin":y_min, "xmax":x_max, "ymax":y_max}')
    return {"xmin": x_min, "ymin": y_min, "xmax": x_max, "ymax": y_max}


NET, DEVICE = None, None


def init_model(model_path=None, device="cuda:0"):
    global NET, DEVICE
    if NET is None:
        if model_path is None:
            model_path = "weights/sam_hq_vit_l.pth"
        t = Timer(f"加载模型:{model_path}")
        if torch.cuda.is_available():
            device = torch.device(device=device)
        else:
            device = torch.device("cpu")
        from segment_anything_hq import sam_model_registry
        model_type = "vit_l"
        sam = sam_model_registry[model_type](checkpoint=model_path)
        log.info(t)
        NET = sam.to(device)
        DEVICE = device


def unique_mask_color(prev_mask):
    t = Timer("unique_color")
    unique_colors = np.unique(
        prev_mask.reshape(-1, prev_mask.shape[2]), axis=0)
    log.info(t("Mask颜色去重"))
    if unique_colors.shape[0] not in [1, 2]:
        raise ValueError("Mask异常，只能含有一个或2个颜色！")
    # 前景色
    if unique_colors.shape[0] == 2:
        f_rgb = unique_colors[unique_colors.sum(
            axis=1) > 0][0].astype(np.uint8)
    else:
        f_rgb = np.array([0, 255, 0], dtype=np.uint8)
    # 背景色
    b_rgb = np.array([0, 0, 0], dtype=np.uint8)
    return f_rgb, b_rgb


def get_suggest_color(img_rgb):
    img_color = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    selected_color = img_color.reshape(-1, 3).mean(axis=0)
    hue = int(selected_color[0])
    hue_new = hue + 90 if hue < 90 else hue - 90
    hsv = [hue_new, 255, 255]
    rgb = list(cv2.cvtColor(np.array([[hsv]]).astype(
        np.uint8), cv2.COLOR_HSV2RGB).reshape(-1))
    return rgb


def check_result_path(result_path):
    p = Path(result_path)
    if p.exists():
        raise Exception("目标路径[{}]已存在文件".format(result_path))
    if not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    return p


class ModelWrapper:
    def __init__(self):
        global NET, DEVICE
        if NET:
            self.net = NET
            self.device = DEVICE
        else:
            raise Exception("模型未加载！")

    def predict(self, body_params):
        t = Timer("Wrapper内部")
        # parse request body params

        result_path = body_params["resultPath"]
        check_result_path(result_path=result_path)
        result_prob_path = result_path+".npy"

        image_path = body_params["imgPath"]
        image = check_image(image_path)

        if body_params.get("preMask", None) in [None, ""]:
            prev_mask = np.zeros(image.shape[:2]+(4,))
            prev_prob = None
        else:
            prev_mask = check_image(body_params["preMask"], 4)
            prev_prob_path = body_params["preMask"]+".npy"
            if Path(prev_prob_path).exists():
                prev_prob = np.load(prev_prob_path)
                log.info("=== mask_input loaded from: {}".format(prev_prob_path))
            else:
                prev_prob = None

        log.info(t("读图"))
        
        if body_params.get("box", None) in [None, ""]:
            clicks = None
            input_point = None
            input_label = None
        else:
            clicks = [(float(c["x"]), float(c["y"]), c["is_positive"]) for c in body_params["click"]]
            clicks = np.array(clicks)
            input_point = clicks[:, :2]
            input_label = clicks[:, 2]

        if image.shape[:2] != prev_mask.shape[:2]:
            raise ValueError("图片与Mask大小不匹配！")

        if body_params.get("box", None) in [None, ""]:
            box = None
        else:
            box = body_params["box"]
            img_h, img_w = image.shape[:2]
            box_h, box_w = box['ymax'] - box['ymin'] + 1, box['xmax'] - box['xmin'] + 1
            if box_h == img_h and box_w == img_w:
                box = None
            else:
                # min max 按照前闭后闭
                assert box["xmin"] >= 0 and box["xmin"] < img_w,  "xmin超出图片范围[{},{}]".format(
                    0, img_w-1)
                assert box["xmax"] >= 0 and box["xmax"] < img_w,  "xmax超出图片范围[{},{}]".format(
                    0, img_w-1)
                assert box["ymin"] >= 0 and box["ymin"] < img_h,  "ymin超出图片范围[{},{}]".format(
                    0, img_h-1)
                assert box["ymax"] >= 0 and box["ymax"] < img_h,  "ymax超出图片范围[{},{}]".format(
                    0, img_h-1)

                box = np.array([box['xmin'], box['ymin'], box['xmax'], box['ymax']])

        from segment_anything_hq import SamPredictor
        predictor = SamPredictor(self.net)
        log.info(t("初始化Predictor"))

        import hashlib
        # 计算图像的hash值作为唯一key
        image_hash = hashlib.md5(image).hexdigest()

        
        # if image_hash in cache.keys():
        #     print(image_hash)
        #     print(cache[image_hash][0].shape)
        #     print(cache[image_hash][0].dtype)
        #     for meta in cache[image_hash][1]:
        #         print(meta.shape)
        predictor.set_image(image, cache, image_hash)
        
        log.info(t("调用Image Encoder"))
        
        

        """
        Returns:
          (np.ndarray): The output masks in CxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (np.ndarray): An array of length C containing the model's
            predictions for the quality of each mask.
          (np.ndarray): An array of shape CxHxW, where C is the number
            of masks and H=W=256. These low resolution logits can be passed to
            a subsequent iteration as mask input.
        """
        # 需要上游保证每次请求中至少出现box或者point的prompt
        masks, scores, low_res_masks_np = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=box,
            mask_input=prev_prob,
            multimask_output=False,
            hq_token_only=False,
        )
        log.info(t("调用Predictor"))

        mask_color = body_params["maskColor"]
        mask_clip = masks[0]
        if mask_color["R"] == -1 or mask_color["G"] == -1 or mask_color["B"] == -1:
            rgb = get_suggest_color(image[mask_clip].reshape(-1, 1, 3))
            f_rgba = np.array(rgb + [mask_color["A"]], dtype=np.uint8)
        else:
            f_rgba = np.array([mask_color["R"], mask_color["G"],
                              mask_color["B"], mask_color["A"]], dtype=np.uint8)
        b_rgba = np.array([0, 0, 0, 0], dtype=np.uint8)
        log.info(t("处理颜色"))

        pred_clip = np.zeros(image.shape[:2]+(4,), dtype=np.uint8)
        pred_clip[:, :] = b_rgba
        
        # 将box和mask取交集，box之外的部分都置为False
        if box is not None:
            xmin, ymin, xmax, ymax = box
            mask_clip[0:ymin, :] = False
            mask_clip[ymax:, :] = False
            mask_clip[ymin:ymax, 0:xmin] = False
            mask_clip[ymin:ymax, xmax:] = False

        pred_clip[mask_clip] = f_rgba

        pred = pred_clip

        if pred[:, :, 3].reshape(-1).sum() == 0:
            log.info("没有预测出来像素,使用上个mask")
            pred = prev_mask.astype(np.uint8)
            cv2.imwrite(result_path, pred)
        else:
            pred = pred[:, :, [2, 1, 0, 3]]
            cv2.imwrite(result_path, pred)
            prev_prob = low_res_masks_np

        np.save(result_prob_path, prev_prob)

        log.info("=== Prob saved to: {}".format(result_prob_path))
        log.info("=== masks scores<--->{}".format(scores))
        log.info(t)
        del predictor
        # 开gc跟不开效果一样
        # import gc
        # gc.collect
        torch.cuda.empty_cache()
        bndbox = get_bndbox(pred)

        return result_path, list(f_rgba), bndbox


def error_1(msg):
    return {"code": -1, "msg": f"ERROR: {msg}"}


def success(img_path_and_rgba, msg="ok"):
    img_path, rgba, bndbox = img_path_and_rgba
    return {"code": 0, "data": {"mask": img_path, "bndbox": bndbox, "maskColor": {"R": int(rgba[0]), "G": int(rgba[1]), "B": int(rgba[2]), "A": (int(rgba[3]))}}, "msg": msg}


class SegMagic(RequestHandler):
    def initialize(self):
        self.model = ModelWrapper()

    def post(self):
        t = Timer("Post全程")
        right_type = "application/json"
        content_type = self.request.headers.get("Content-Type", "")
        if content_type == right_type:
            raw_body = self.request.body
            try:
                body_dict = json.loads(raw_body)
                log.info(f"body_dict: {body_dict}")
            except Exception as e:
                log.exception(f"RAW body to json failed:\n{raw_body}")
                self.write(error_1(f"Can't decode json content."))
                return
            else:
                try:
                    mask_path = self.model.predict(body_dict)
                    log.info(t)
                    self.write(success(mask_path))
                except Exception as e:
                    log.exception("Failed to predict.")
                    import traceback
                    msg = f"Failed to predict:\n{traceback.format_exc()}"
                    self.write(error_1(msg))
        else:
            msg = f"Wrong Content-Type: {content_type}. Only accept: {right_type}"
            log.error(msg)
            self.write(error_1(msg))
            return


class Liveness(RequestHandler):
    def get(self):
        self.write(
            f"Liveness: {datetime.strftime(datetime.today(), '%Y-%m-%d %H:%M:%S')}")


class Readiness(RequestHandler):
    def get(self):
        self.write(
            f"Readiness: {datetime.strftime(datetime.today(), '%Y-%m-%d %H:%M:%S')}")


def make_app():
    return Application([
        url(r"/", MainHandler),
        url(r"/seg/magic", SegMagic),
        url(r"/actuator/health/liveness", Liveness),
        url(r"/actuator/health/readiness", Readiness)
    ])


async def main():
    app = make_app()
    app.listen(8888)
    shutdown_event = asyncio.Event()
    await shutdown_event.wait()

init_model()

if __name__ == "__main__":
    asyncio.run(main())
