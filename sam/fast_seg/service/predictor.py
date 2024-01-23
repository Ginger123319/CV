import torch
import torch.nn.functional as F
from torchvision import transforms

from isegm.inference.transforms import AddHorizontalFlip, SigmoidForPred, LimitLongestSide
from copy import deepcopy
from .utils import Timer, log


class Click:
    def __init__(self, is_positive, coords, indx=None):
        self.is_positive = is_positive
        self.coords = coords
        self.indx = indx

    @property
    def coords_and_indx(self):
        return (*self.coords, self.indx)

    def copy(self, **kwargs):
        self_copy = deepcopy(self)
        for k, v in kwargs.items():
            setattr(self_copy, k, v)
        return self_copy


class BasePredictor(object):
    def __init__(self, model, device,
                 net_clicks_limit=None,
                 with_flip=False,
                 zoom_in=None,
                 max_size=None,
                 **kwargs):
        self.with_flip = with_flip
        self.net_clicks_limit = net_clicks_limit
        self.device = device
        self.zoom_in = zoom_in
        self.prev_prediction = None
        self.model_indx = 0
        self.net_state_dict = None

        self.net = model

        self.to_tensor = transforms.ToTensor()

        self.transforms = [zoom_in] if zoom_in is not None else []
        if max_size is not None:
            self.transforms.append(LimitLongestSide(max_size=max_size))
        self.transforms.append(SigmoidForPred())
        if with_flip:
            self.transforms.append(AddHorizontalFlip())

    def get_prediction(self, image, clicks, prev_mask=None):
        # 这里要转成Click
        t = Timer("Predictor内部")
        clicks_list = [Click(is_positive=_is_positive, coords=(_y, _x)) for _x, _y, _is_positive in clicks]
        for i in range(len(clicks_list)):
            clicks_list[i].indx = i

        input_image = self.to_tensor(image)
        for transform in self.transforms:
            transform.reset()
        input_image = input_image.to(self.device).unsqueeze(0)
        prev_mask = torch.tensor(prev_mask).to(self.device)[None, None]
        input_image = torch.cat((input_image, prev_mask), dim=1)
        log.info(t("prepare input"))
        
        image_nd, clicks_lists, is_image_changed = self.apply_transforms(
            input_image, [clicks_list]
        )
        log.info(t("apply transform"))

        pred_logits = self._get_prediction(image_nd, clicks_lists, is_image_changed)
        log.info(t("net forward"))
        prediction = F.interpolate(pred_logits, mode='bilinear', align_corners=True,
                                   size=image_nd.size()[2:])

        log.info(t("interpolate"))
        for trans in reversed(self.transforms):
            prediction = trans.inv_transform(prediction)
        log.info(t("reverse transform"))
        
        #if self.zoom_in is not None and self.zoom_in.check_possible_recalculation():
        #   return self.get_prediction(clicks)

        #self.prev_prediction = (prediction>0.5).type_as(prediction)
        # self.prev_prediction = prediction
        prediction = prediction.cpu().numpy()[0, 0]
        log.info(t)
        return prediction

    def _get_prediction(self, image_nd, clicks_lists, is_image_changed):
        points_nd = self.get_points_nd(clicks_lists)
        return self.net(image_nd, points_nd)['instances']

    def _get_transform_states(self):
        return [x.get_state() for x in self.transforms]

    def _set_transform_states(self, states):
        assert len(states) == len(self.transforms)
        for state, transform in zip(states, self.transforms):
            transform.set_state(state)

    def apply_transforms(self, image_nd, clicks_lists):
        is_image_changed = False
        for t in self.transforms:
            image_nd, clicks_lists = t.transform(image_nd, clicks_lists)
            is_image_changed |= t.image_changed

        return image_nd, clicks_lists, is_image_changed

    def get_points_nd(self, clicks_lists):
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list) for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list, num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [click.coords_and_indx for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [click.coords_and_indx for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)

        return torch.tensor(total_clicks, device=self.device)

    def get_states(self):
        return {
            'transform_states': self._get_transform_states(),
            'prev_prediction': self.prev_prediction.clone()
        }

    def set_states(self, states):
        self._set_transform_states(states['transform_states'])
        self.prev_prediction = states['prev_prediction']

