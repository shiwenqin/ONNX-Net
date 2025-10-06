import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchsort
from transformers import Trainer

class HuberTrainer(Trainer):
    " Use Huber loss for regression tasks. "
    def __init__(self, *args, delta=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.HuberLoss(delta=delta)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits.squeeze(-1)  # shape: (batch_size)
        if logits is None:
            logits = outputs[0]

        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        labels = labels.to(logits.dtype)
        loss = self.loss(logits, labels)

        return (loss, outputs) if return_outputs else loss
    
class MSECapTrainer(Trainer):
    " Use MSE loss with a cap for regression tasks. "
    def __init__(self, *args, cap=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.cap = cap
        self.loss = nn.MSELoss()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits.squeeze(-1)  # shape: (batch_size)
        if logits is None:
            logits = outputs[0]

        if logits.dim() > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)
        labels = labels.to(logits.dtype)
        loss = self.loss(logits, labels)
        loss = torch.clamp(loss, max=self.cap)

        return (loss, outputs) if return_outputs else loss
    
class PWRTrainer(Trainer):
    " Use Pairwise Ranking loss for regression tasks. "
    def __init__(self, *args, compare_threshold=0.0, max_compare_ratio=4, margin=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.compare_threshold = compare_threshold
        self.max_compare_ratio = max_compare_ratio
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits.squeeze(-1)

        t = labels.to(logits.device, dtype=logits.dtype)
        B = t.size(0)

        diff = t[:, None] - t[None, :]
        mask = torch.triu(diff.abs() > self.compare_threshold, diagonal=1)
        i, j = mask.nonzero(as_tuple=True)
        m = i.numel()

        if m == 0:
            loss = (logits * 0).sum() 
            return (loss, outputs) if return_outputs else loss
        
        n_max_pairs = min(m, self.max_compare_ratio * B)
        if m > n_max_pairs:
            perm = torch.randperm(m, device=logits.device)[:n_max_pairs]
            i, j = i[perm], j[perm]

        s_i, s_j = logits[i], logits[j]
        y = torch.sign(t[i] - t[j]).to(dtype=logits.dtype)
        y = torch.where(y == 0, torch.ones_like(y), y)

        loss = self.loss(s_i, s_j, y)

        return (loss, outputs) if return_outputs else loss
    
class PairwiseLogTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits.squeeze(-1)
        t = labels.to(logits.device, dtype=logits.dtype)

        diff = logits.unsqueeze(1) - logits.unsqueeze(0)
        pair_mask = t.unsqueeze(1) > t.unsqueeze(0)

        if pair_mask.any():
            loss_pairs = F.softplus(-diff)[pair_mask]
            loss = loss_pairs.mean()
        else:
            loss = (logits * 0).sum()

        return (loss, outputs) if return_outputs else loss
    
class SoftmaxListwiseTrainer(Trainer):

    def __init__(self, *args, tau: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.tau = float(tau)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits.squeeze(-1)
        t = labels.to(logits.device, dtype=logits.dtype)

        y_prob = F.softmax(t / self.tau, dim=0)

        log_p = F.log_softmax(logits, dim=0)
        loss = -(y_prob * log_p).sum()
        # norm = labels.sum().clamp_min(1e-8)
        # loss = ce / norm

        return (loss, outputs) if return_outputs else loss
    
class Poly1ListwiseTrainer(Trainer):

    def __init__(self, *args, epsilon: float = 1.0, tau: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = float(epsilon)
        self.tau = float(tau)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits.squeeze(-1)
        t = labels.to(logits.device, dtype=logits.dtype)

        y_prob = F.softmax(t / self.tau, dim=0)

        log_p = F.log_softmax(logits, dim=0)
        p = torch.exp(log_p)
        ce = -(y_prob * log_p).sum()
        poly = (y_prob * (1 - p)).sum()
        #norm = labels.sum().clamp_min(1e-8)
        #loss = (ce + self.epsilon * poly) / norm
        loss = ce + self.epsilon * poly

        return (loss, outputs) if return_outputs else loss
    
# class SpearmanTrainer(Trainer):
#     " Use Spearman correlation as loss for regression tasks. "
#     def __init__(self, *args, tau=1.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.tau = tau

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)

#         logits = outputs.logits.squeeze(-1)

#         y_pred = logits.view(-1).to(torch.float32)
#         y_true = labels.to(logits.device).view(-1).to(torch.float32)

#         r_pred = torchsort.soft_rank(y_pred[None, :].contiguous(), regularization_strength=self.tau)
#         r_true = torchsort.soft_rank(y_true[None, :].contiguous(), regularization_strength=self.tau)

#         r_pred = (r_pred - r_pred.mean()) / (r_pred.std() + 1e-8)
#         r_true = (r_true - r_true.mean()) / (r_true.std() + 1e-8)

#         loss = 1 - (r_pred * r_true).mean()

#         return (loss, outputs) if return_outputs else loss

class PWRMiningTrainer(Trainer):
    " Use Pairwise Ranking loss with hard mining for regression tasks. "
    def __init__(self, *args, 
                 compare_threshold=0.0, 
                 max_compare_ratio=4, 
                 margin=0.1, 
                 weight_mode='exp', 
                 max_pair_weight=None,
                 mining_mode='topk',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.compare_threshold = compare_threshold
        self.max_compare_ratio = max_compare_ratio
        self.margin = margin
        self.loss = nn.MarginRankingLoss(margin=margin, reduction='none')
        self.weight_mode = weight_mode
        self.tau = 0.5
        self.eps = 1e-8
        self.max_pair_weight = max_pair_weight
        self.mining_mode = mining_mode

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        logits = outputs.logits.squeeze(-1)

        t = labels.to(logits.device, dtype=logits.dtype)
        B = t.size(0)

        diff = t[:, None] - t[None, :]
        mask = torch.triu(diff.abs() > self.compare_threshold, diagonal=1)
        i, j = mask.nonzero(as_tuple=True)
        m = i.numel()

        if m == 0:
            loss = (logits * 0).sum() 
            return (loss, outputs) if return_outputs else loss
        
        n_max_pairs = min(m, self.max_compare_ratio * B)

        s_i, s_j = logits[i], logits[j]
        y = torch.sign(t[i] - t[j]).to(dtype=logits.dtype)
        y = torch.where(y == 0, torch.ones_like(y), y)

        loss = self.loss(s_i, s_j, y)
        gaps = diff.abs()[i, j]

        if self.weight_mode == 'exp':
            w = torch.exp(-gaps / max(self.tau, self.eps))
        elif self.weight_mode == 'inverse':
            w = 1.0 / (gaps + self.eps)
        else:
            w = torch.ones_like(gaps)

        if self.max_pair_weight is not None:
            w = torch.clamp(w, max=self.max_pair_weight)

        if m > n_max_pairs or self.mining_mode != 'none':

            target_k = n_max_pairs
            if self.mining_mode == 'topk':
                k =  min(target_k, m)
                if k < m:
                    idx = torch.topk(loss, k=k, largest=True).indices
                    loss = loss[idx]
                    w = w[idx]
            else:
                if m > target_k:
                    perm = torch.randperm(m, device=logits.device)[:target_k]
                    loss = loss[perm]
                    w = w[perm]

        loss = (loss * w).sum() / (w.sum() + self.eps)

        return (loss, outputs) if return_outputs else loss

# class PlackettTrainer(Trainer):
#     " Use Plackett-Luce loss for regression tasks. "
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)

#         logits = outputs.logits.squeeze(-1)

#         y_pred = logits.view(-1)
#         y_true = labels.to(logits.device).view(-1)

#         sorted_pred = torch.argsort(y_pred, descending=True)
#         sorted_true = y_true[sorted_pred]

#         loss = 0.0
#         for i in range(len(y_true)):
#             denominator = torch.logsumexp(sorted_true[i:], dim=0)
#             loss += (denominator - sorted_true[i])
#         loss = loss / len(y_true)

#         return (loss, outputs) if return_outputs else loss
    
# class ApproxRankTrainer(Trainer):
#     " Use Approximate Ranking loss for regression tasks. "
#     def __init__(self, *args, alpha=10.0, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.alpha = alpha

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
#         labels = inputs.pop("labels")
#         outputs = model(**inputs)
        
#         logits = outputs.logits.squeeze(-1)

#         y_pred = logits.unsqueeze(0)
#         y_true = labels.to(logits.device).unsqueeze(0)

#         pairwise_diff = y_pred.T - y_pred
#         pairwise_labels = (y_true.T - y_true)

#         smooth_rank = torch.sigmoid(self.alpha * pairwise_diff)

#         loss = nn.functional.binary_cross_entropy(smooth_rank, pairwise_labels)

#         return (loss, outputs) if return_outputs else loss