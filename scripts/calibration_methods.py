"""Explicit multi-label adaptations of the PPT's training interventions.

No softmax across diseases. SPA: Cheng & Vasconcelos CVPR 2024, Eq.16,
19-21. Its probability link is derived from the partial-loss derivatives;
see the sign discussion in the study guide. MDCA uses sigmoid marginals.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F


DEFAULT_METHOD = dict(loss="bce", scope="full", uncertain="original", smoothing=0.05,
                      focal_gamma=2.0, mdca_weight=1.0, mixup_alpha=0.2,
                      soft_low=0.55, soft_high=0.85, partial_blocks=2,
                      lora_rank=8, lora_alpha=16.0, spa_zeta=5.0,
                      spa_k=3.0, spa_b=1.0, lpr_weight=0.1)
LOSSES = ("bce", "label_smoothing", "focal", "mdca", "mixup", "spa_lpr")
SCOPES = ("full", "head", "partial", "lora")
UNCERTAINTIES = ("original", "ones", "zeros", "ignore", "soft")


def method_config(values=None):
    values = values or {}
    if set(values) - set(DEFAULT_METHOD):
        raise ValueError("Unknown method settings: " + str(sorted(set(values) - set(DEFAULT_METHOD))))
    cfg = dict(DEFAULT_METHOD, **values)
    if cfg["loss"] not in LOSSES or cfg["scope"] not in SCOPES or cfg["uncertain"] not in UNCERTAINTIES:
        raise ValueError("Unknown loss/scope/uncertain policy")
    for key, value in cfg.items():
        if isinstance(value, (float, int)) and (isinstance(value, bool) or not math.isfinite(value)):
            raise ValueError("Non-finite method parameter: " + key)
    if not 0 <= cfg["smoothing"] < 1 or not 0 <= cfg["soft_low"] <= cfg["soft_high"] <= 1:
        raise ValueError("Invalid label smoothing or uncertain soft range")
    if any(cfg[k] < 0 for k in ("focal_gamma", "mdca_weight", "lpr_weight")):
        raise ValueError("Loss weights/gamma must be nonnegative")
    if any(cfg[k] <= 0 for k in ("mixup_alpha", "lora_alpha", "spa_zeta", "spa_k")):
        raise ValueError("Scales must be positive")
    if any(type(cfg[k]) is not int or cfg[k] < 1 for k in ("partial_blocks", "lora_rank")):
        raise ValueError("Block count and LoRA rank must be positive integers")
    return cfg


def probability_logits(logits, cfg):
    """Return Bernoulli log odds, including SPA's increasing inverse link.

    Positive partial loss = softplus(-v); negative = softplus(k*(v-b))/zeta.
    eta/(1-eta) = negative_derivative / (-positive_derivative).
    This also reduces exactly to v for k=zeta=1,b=0.
    """
    logits = logits.float()
    if cfg["loss"] == "spa_lpr":
        return (math.log(cfg["spa_k"] / cfg["spa_zeta"])
                + F.logsigmoid(cfg["spa_k"] * (logits - cfg["spa_b"]))
                - F.logsigmoid(-logits))
    return logits


def masked_mean(values, weights):
    # A fully masked batch must still participate in DDP with zero gradients.
    weights = weights.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1)


def objective(logits, targets, mask, cfg):
    logits, targets = logits.float(), targets.float()
    loss = cfg["loss"]
    if loss == "label_smoothing":
        targets = targets * (1 - cfg["smoothing"]) + cfg["smoothing"] / 2
    if loss == "spa_lpr":
        elements = targets * F.softplus(-logits) + (1 - targets) * (
            F.softplus(cfg["spa_k"] * (logits - cfg["spa_b"])) / cfg["spa_zeta"])
    elif loss == "focal":
        p = logits.sigmoid()
        # Weighted positive/negative partial losses also support soft U labels.
        elements = (targets * (1 - p).pow(cfg["focal_gamma"]) * F.softplus(-logits)
                    + (1 - targets) * p.pow(cfg["focal_gamma"]) * F.softplus(logits))
    else:
        elements = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    result = masked_mean(elements, mask)
    if loss == "mdca":
        weights = mask.float()
        counts = weights.sum(0)
        difference = ((logits.sigmoid() - targets) * weights).sum(0) / counts.clamp_min(1)
        regularizer = (difference.abs() * (counts > 0)).sum() / (counts > 0).sum().clamp_min(1)
        result = result + cfg["mdca_weight"] * regularizer
    if loss == "spa_lpr":
        log_odds = probability_logits(logits, cfg)
        # P(i positive | i != j) = sigmoid(log_odds_i - log_odds_j).
        pair_logits = log_odds[:, :, None] - log_odds[:, None, :]
        pair_weights = (targets[:, :, None] * (1 - targets[:, None, :])
                        * mask[:, :, None] * mask[:, None, :])
        off_diagonal = 1 - torch.eye(logits.shape[1], device=logits.device)
        pair_loss = (F.softplus(-pair_logits) * pair_weights * off_diagonal).sum((1, 2))
        # Eq.20 normalization; soft U targets use expected independent pair labels.
        regularizer = pair_loss.mean() * 2 / (logits.shape[1] * (logits.shape[1] - 1))
        result = result + cfg["lpr_weight"] * regularizer
    return result


def mix_batch(images, targets, mask, alpha):
    """Mix valid-label mass, preserving U-Ignore masks instead of filling unknowns."""
    beta = torch.distributions.Beta(alpha, alpha)
    ratio = beta.sample().to(images.device)
    permutation = torch.randperm(len(images), device=images.device)
    left, right = ratio * mask.float(), (1 - ratio) * mask[permutation].float()
    mass = left + right
    mixed_targets = (left * targets + right * targets[permutation]) / mass.clamp_min(1e-12)
    return ratio * images + (1 - ratio) * images[permutation], mixed_targets, mass


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.lora_a = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_b = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    @property
    def weight(self):
        # EVA's Q/K/V path calls F.linear(..., projection.weight) directly.
        return self.base.weight + (self.lora_b @ self.lora_a) * self.scale

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        return self.base(inputs) + F.linear(F.linear(inputs, self.lora_a), self.lora_b) * self.scale


def configure_scope(model, cfg):
    scope = cfg["scope"]
    if scope == "full":
        return
    model.requires_grad_(False)
    model.head.requires_grad_(True)
    if scope != "head" and getattr(model, "fc_norm", None) is not None:
        model.fc_norm.requires_grad_(True)
    if scope == "partial":
        if cfg["partial_blocks"] > len(model.blocks):
            raise ValueError("partial_blocks exceeds model depth")
        for block in model.blocks[-cfg["partial_blocks"]:]:
            block.requires_grad_(True)
    if scope == "lora":
        count = 0
        for block in model.blocks:
            for name in ("qkv", "q_proj", "k_proj", "v_proj", "proj"):
                layer = getattr(block.attn, name, None)
                if isinstance(layer, nn.Linear):
                    setattr(block.attn, name, LoRALinear(layer, cfg["lora_rank"], cfg["lora_alpha"]))
                    count += 1
        if not count:
            raise ValueError("No EVA attention projections found for LoRA")


def training_mode(model, cfg):
    bare = model.module if hasattr(model, "module") else model
    model.train()
    if cfg["scope"] != "full":
        # Frozen features stay deterministic; train dropout/stochastic depth only
        # in explicitly trainable blocks. LoRA has no adapter dropout.
        bare.eval()
        bare.head.train()
        if getattr(bare, "fc_norm", None) is not None and cfg["scope"] != "head":
            bare.fc_norm.train()
        if cfg["scope"] == "partial":
            for block in bare.blocks[-cfg["partial_blocks"]:]:
                block.train()
