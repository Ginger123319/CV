from module import *
import cfg


class Detector(torch.nn.Module):

    def __init__(self):
        super(Detector, self).__init__()

        self.net = Darknet53()
        self.net.eval()

    def forward(self, input, thresh, anchors):
        output_13, output_26, output_52 = self.net(input)

        idxs_13, vecs_13 = self._filter(output_13, thresh)
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])

        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])

        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])

        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        # 筛选IOU大于阈值的输出数据
        output = output.permute(0, 2, 3, 1)
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        # output:N,H,W,3,15
        # mask:N,H,W,3
        mask = output[..., 0] > thresh
        # 取出所有IOU通道上为True的位置索引
        # 形状为N,V
        idxs = mask.nonzero()
        # print(idxs.shape)
        # print(idxs)
        # exit()
        # 取出IOU大于阈值的输出中的6个目标值iou cx cy w h cls
        vecs = output[mask]
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):
        anchors = torch.Tensor(anchors)
        # idxs的形状为NV，其中V代表NHW3
        n = idxs[:, 0]  # 取得V中的N，表示图片的索引
        a = idxs[:, 3]  # 取得V中的3，即3个建议框中取几个建议框
        # idxs[:, 1]是H即中心点坐标y在13*13特征图上的索引值， vecs[:, 2]即y的偏移量
        # （索引值+偏移量）*缩放比例（32）=原始图片上的中心点坐标
        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # 原图的中心点y
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # 原图的中心点x
        # 根据建议框计算输出框的宽高
        w = anchors[a, 0] * torch.exp(vecs[:, 3])
        h = anchors[a, 1] * torch.exp(vecs[:, 4])

        return torch.stack([n.float(), cx, cy, w, h], dim=1)


if __name__ == '__main__':
    detector = Detector()
    y = detector(torch.randn(3, 3, 416, 416), 0.01, cfg.ANCHORS_GROUP)
    print(y.shape)
