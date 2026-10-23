import torch

# FusedLayerNorm과 파라미터 이름(weight, bias)·인자가 같아 체크포인트 그대로 로드 가능
FusedLayerNorm = torch.nn.LayerNorm
