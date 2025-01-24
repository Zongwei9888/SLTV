import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import LogNormal
from params import args
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

def ziln_loss(labels, logits, class_threshold=0, with_mask=False):
    # [batch_size, 1]
    logits = logits
    labels = labels
    labels = labels.view(-1, 1)

    # 分类标签
    positive = (labels > class_threshold).float()

    # logits p, u, sigma, mask


    positive_logits = logits[..., :1]
    # [batch_size, ]
    classification_loss = F.binary_cross_entropy_with_logits(
        input=positive_logits, target=positive, reduction='none')

    loc = logits[..., 1:2]
    # softplus(x) = log(1+exp(x))
    scale = torch.maximum(
        torch.nn.functional.softplus(logits[..., 2:3]),
        torch.sqrt(torch.tensor(1e-7)))
    # [batch_size, 1]
    safe_labels = positive * labels + (1 - positive) * torch.ones_like(labels)
    # [batch_size, ]
    regression_loss = -torch.mean(positive * LogNormal(loc=loc, scale=scale).log_prob(safe_labels), dim=-1)

    # if with_mask:
    #     mask = logits[..., 3:].squeeze()
    #     classification_loss = classification_loss * mask
    #     regression_loss = regression_loss * mask

    return torch.mean(classification_loss + regression_loss)


def ziln_pred(logits, pred_clip_val=0):
    """
    Calculates predicted mean of zero inflated lognormal logits.

    Arguments:
      logits: [batch_size, 3] tensor of logits.

    Returns:
      preds: [batch_size, 1] tensor of predicted mean.
    """
    positive_probs = torch.sigmoid(logits[..., :1])
    loc = logits[..., 1:2]
    # softplus(x) = log(exp(x) + 1)
    scale = F.softplus(logits[..., 2:3])
    preds = (
        positive_probs * torch.exp(
            loc + 0.5 * torch.square(scale)
        )
    )

    # nan 为异常情况, 处理为0
    is_nan = torch.isnan(preds)
    padding_preds = torch.zeros_like(preds)
    preds = torch.where(is_nan, padding_preds, preds)
    if pred_clip_val > 0:
        preds = torch.clamp(preds, 0, pred_clip_val)
    return positive_probs, preds



def js_div(p_output, q_output):
    """
    Function that measures JS divergence between target and output logits:
    """
    KLDivLoss = nn.KLDivLoss(reduction='batchmean')
    log_mean_output = ((p_output + q_output) / 2+1e-8).log()
    return (KLDivLoss(log_mean_output, p_output) + KLDivLoss(log_mean_output, q_output)) / 2

def discrepancy_l1(out1, out2):
    return torch.mean(torch.abs(out1 - out2))

def discrepancy_l2(out1, out2):
    return torch.mean(torch.square(out1 - out2))

