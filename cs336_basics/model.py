from einops import reduce
import torch
from torch import nn
import math

class Linear(nn.Module):
    """Bias-free linear module with the assignment's recommended interface."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        sigma = math.sqrt(2/(in_features+out_features))
        W_init = nn.init.trunc_normal_(torch.empty(size=(out_features,in_features),device=device,dtype=dtype),mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        self.W = nn.Parameter(data=W_init)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1]!=self.W.shape[-1]:
            raise ValueError("feature tensor does not have the correct shape")
        if x.device!=self.W.device:
            raise ValueError("feature tensor not on the same device as linear layer params")
        if x.dtype!=self.W.dtype:
            raise ValueError("feature tensor dtype mismatch with linear layer parameter dtype")
        return torch.einsum('...i, oi -> ...o',x,self.W)


class Embedding(nn.Module):
    """Embedding module with the assignment's recommended interface."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        W_init = torch.nn.init.trunc_normal_(torch.empty(size=(num_embeddings,embedding_dim),device=device,dtype=dtype),mean=0,std=1,a=-3,b=3)
        self.W_embd = nn.Parameter(data=W_init)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        
        return self.W_embd[token_ids]


class RMSNorm(nn.Module):
    """RMSNorm module with the assignment's recommended interface."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.gain = torch.ones(size=(d_model,),dtype=dtype,device=device)
        self.eps=eps
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype=x.dtype
        x=x.to(dtype=torch.float32)
        x_norm =torch.sqrt(torch.mean(torch.square(x),dim=-1,keepdim=True)+self.eps)
        result = (x/x_norm)*self.gain
        return result.to(dtype=in_dtype)
        
