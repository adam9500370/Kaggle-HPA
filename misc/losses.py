import torch
import torch.nn.functional as F


# input/target size: (n, c)
def f_beta_loss(input, target, size_average=True, beta=1, eps=1e-6):
    p = F.sigmoid(input)
    l = target

    num_pos_pred = p.sum(1) + eps # TP + FP
    num_pos_true = l.sum(1) + eps # TP + FN

    tp = (l * p).sum(1)
    precision = tp / num_pos_pred
    recall = tp / num_pos_true

    fs = (1 + beta * beta) * precision * recall / (beta * beta * precision + recall + eps)
    loss = 1 - fs

    loss = loss.mean() if size_average else loss.sum()
    return loss


# input/target size: (n, c)
def binary_focal_loss_with_logits(input, target, pos_weight=None, size_average=True, alpha=0.25, gamma=2.0):
    if not (target.size() == input.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(target.size(), input.size()))

    max_val = (-input).clamp(min=0)
    loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log() # Cross Entropy (CE)
    if pos_weight is not None:
        loss = (pos_weight.unsqueeze(0) * target + (1. - target)) * loss

    a_t = alpha * target + (1-alpha) * (1-target)
    loss = a_t * loss # alpha-balanced CE

    invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
    loss = (invprobs * gamma).exp() * loss # Focal Loss (FL)

    loss = loss.mean() if size_average else loss.sum()
    return loss
