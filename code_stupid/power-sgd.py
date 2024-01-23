import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
# 梯度压缩，减少通信量，用于分布式训练中加快训练速度
state = powerSGD.PowerSGDState(
    process_group=None,
    matrix_approximation_rank=1,
    start_powerSGD_iter=2,
    # 避免除0的错误
    orthogonalization_epsilon=1e-8,
    use_error_feedback=True, warm_start=True, random_seed=0
)
self.net.register_comm_hook(state, powerSGD.powerSGD_hook)