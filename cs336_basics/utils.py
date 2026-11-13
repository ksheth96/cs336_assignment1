from collections.abc import Iterable
import math

import torch

def softmax(x: torch.tensor, i: int) -> torch.tensor:
    x_exp = torch.exp(x - torch.max(input=x,dim=i,keepdim=True).values)
    return x_exp/torch.sum(x_exp,dim=i,keepdim=True)

def silu(x: torch.tensor) -> torch.tensor:
    return torch.sigmoid(x) * x

def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply scaled dot-product attention with arbitrary leading batch dimensions.

    Args:
        q: Query tensor with shape (..., queries, d_k).
        k: Key tensor with shape (..., keys, d_k).
        v: Value tensor with shape (..., keys, d_v).
        mask: Optional boolean tensor broadcastable to (..., queries, keys).
            True allows attention; False excludes that position.

    Returns:
        Attention output with shape (..., queries, d_v).
    """
    qk = (q @ k.transpose(-2,-1))/math.sqrt(q.shape[-1])
    if mask is None:
        mask = torch.ones(size=qk.shape).bool()
    
    return softmax(x=qk.masked_fill_(mask=~mask,value=-torch.inf), i=-1) @ v    


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute numerically stable mean cross-entropy from unnormalized logits.

    Args:
        inputs: Floating-point logits with shape (..., vocab_size).
        targets: Integer class IDs with shape matching the leading dimensions
            of inputs. Each ID must be between 0 and vocab_size - 1.

    Returns:
        A scalar tensor containing the loss averaged across all batch-like
        dimensions.
    """
    max_logits = torch.max(inputs,dim=-1,keepdim=True).values
    return torch.mean(max_logits - torch.gather(inputs,dim=-1,index=targets.unsqueeze(-1)) + torch.log(torch.sum(torch.exp(inputs-max_logits),dim=-1,keepdim=True)))


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it*max_learning_rate/warmup_iters
    elif warmup_iters <= it <= cosine_cycle_iters:
        return min_learning_rate + (1/2)*(max_learning_rate-min_learning_rate)*(1+math.cos(math.pi*(it-warmup_iters)/(cosine_cycle_iters-warmup_iters)))
    else:
        return min_learning_rate


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    norm = 0
    grad_ref = []
    for grad in parameters:
        if grad.grad is not None:
            norm += torch.norm(grad.grad)**2
            grad_ref.append(grad)
    norm = math.sqrt(norm)
    if norm > max_l2_norm:
            for grad in grad_ref:
                grad.grad *= (max_l2_norm/(norm+1e-6))
  
    
    

