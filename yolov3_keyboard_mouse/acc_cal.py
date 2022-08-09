import torch


def _target_cal(output, is_reshape=True):  # 过滤置信度函数，将置信度合格留下来
    if is_reshape:
        output = output.permute(0, 2, 3, 1)  # 数据形状[N,C,H,W]-->>标签形状[N,H,W,C]，，因此这里通过换轴

        # 通过reshape变换形状 [N,H,W,C]即[N,H,W,45]-->>[N,H,W,3,15]，分成3个建议框每个建议框有15个值
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)
        output = torch.argmax(output[..., -2:], -1)
        result = output
    else:
        result = torch.sum(torch.sum((output[..., -1] > 0), dim=-1) == 3)
    return result


if __name__ == '__main__':
    data = torch.randn(5, 13, 13, 3, 6)
    print(_target_cal(data, False))
