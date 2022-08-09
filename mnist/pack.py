from torch import jit
from net import MnistNet
import torch

if __name__ == '__main__':
    module = MnistNet()
    module.load_state_dict(torch.load(r"checkpoint/18.t"))
    inputs = torch.randn(1, 784)
    traced_script_module = jit.trace(module, inputs)
    traced_script_module.save("mnist.pt")
    output = traced_script_module(torch.ones(1, 784))
    print(output.shape)
