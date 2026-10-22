import math

import torch

def softmax(x: torch.tensor, i: int) -> torch.tensor:
    x_exp = torch.exp(x - torch.max(input=x,dim=i,keepdim=True).values)
    return x_exp/torch.sum(x_exp,dim=i,keepdim=True)


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
    

