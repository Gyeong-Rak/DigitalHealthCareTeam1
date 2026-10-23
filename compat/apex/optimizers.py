import torch


class FusedAdam(torch.optim.AdamW):
    """apex FusedAdam(adam_w_mode=True, 기본값)과 동일한 AdamW. --optimizer fusedlamb일 때만 사용됨."""

    def __init__(self, params, lr=1e-3, bias_correction=True, betas=(0.9, 0.999), eps=1e-8,
                 adam_w_mode=True, weight_decay=0., amsgrad=False, set_grad_none=True):
        super().__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
