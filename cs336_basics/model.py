import torch
from torch import nn
import math

from cs336_basics.utils import scaled_dot_product_attention

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
        


class SwiGLU(nn.Module):
    """Position-wise feed-forward module for the assignment's SwiGLU problem."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = math.ceil(8*d_model/192)*64
        sigma = math.sqrt(2/(d_model+d_ff))
        W_1_init = nn.init.trunc_normal_(torch.empty(size=(d_model,d_ff),device=device,dtype=dtype),mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        W_2_init = nn.init.trunc_normal_(torch.empty(size=(d_ff,d_model),device=device,dtype=dtype),mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        W_3_init = nn.init.trunc_normal_(torch.empty(size=(d_model,d_ff),device=device,dtype=dtype),mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        self.W_1 = nn.Parameter(data=W_1_init)
        self.W_2 = nn.Parameter(data=W_2_init)
        self.W_3 = nn.Parameter(data=W_3_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape (...,d_model), most likely (batch_size,seq_len,d_model)
        # first, silu x
        w1_x = torch.einsum('...i, io -> ...o',x,self.W_1)
        w3_x = torch.einsum('...i, io -> ...o',x,self.W_3)
        return torch.einsum('...o, oi -> ...i', torch.sigmoid(w1_x) * w1_x * w3_x, self.W_2)
        


class RotaryPositionalEmbedding(nn.Module):
    """RoPE module with the assignment's recommended interface."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
    
        seq_len_vector = torch.linspace(start=0,end=max_seq_len-1,steps=max_seq_len,device=device)
        theta_vector = torch.pow(1/theta,(2*torch.linspace(start=1,end=int(d_k/2),steps=int(d_k/2),device=device)-2)/d_k)
        seq_len_theta_cartesian_prod = torch.einsum('i,o -> io',seq_len_vector,theta_vector)
        sin_rope_buffer = torch.sin(seq_len_theta_cartesian_prod)
        cos_rope_buffer = torch.cos(seq_len_theta_cartesian_prod)
        self.register_buffer(name="sin_rope_buffer", tensor=sin_rope_buffer,persistent=False)  
        self.register_buffer(name="cos_rope_buffer", tensor=cos_rope_buffer,persistent=False)  

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        sin_rotations = self.sin_rope_buffer[token_positions]
        cos_rotations = self.cos_rope_buffer[token_positions]
        x_even = x[...,0:x.shape[-1]-1:2]
        x_odd = x[...,1:x.shape[-1]:2]
        output = torch.empty(size=x.shape,device=x.device,dtype=x.dtype)
        output[...,0:x.shape[-1]-1:2] = cos_rotations*x_even-sin_rotations*x_odd
        output[...,1:x.shape[-1]:2] = cos_rotations*x_odd+sin_rotations*x_even
        return output


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention from Section 3.4.5 of the assignment.

    The per-head dimensions satisfy d_k = d_v = d_model / num_heads.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        rope: RotaryPositionalEmbedding | None = None,
    ) -> None:
        """Initialize the module.

        Args:
            d_model: Input and output width; must be divisible by num_heads.
            num_heads: Number of attention heads.
            device: Device for the module's parameters.
            dtype: Data type for the module's parameters.
            rope: Optional preconfigured rotary positional embedding module.
        """
        super().__init__()
        sigma = math.sqrt(1/d_model)
        W_init = nn.init.trunc_normal_(torch.empty(size=(3*d_model,d_model),device=device,dtype=dtype),mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        W_O_init = nn.init.trunc_normal_(torch.empty(size=(d_model,d_model),device=device,dtype=dtype),mean=0,std=sigma,a=-3*sigma,b=3*sigma)
        self.W = nn.Parameter(W_init)
        self.W_O = nn.Parameter(W_O_init)
        self.d_k = int(d_model/num_heads)
        self.num_heads = num_heads
        if rope is not None:
            self.rope = rope
        else:
            self.rope = None


    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply causal self-attention, preventing access to future positions.

        Args:
            x: Input tensor with shape (..., sequence_length, d_model).
            token_positions: Optional integer positions with shape
                (..., sequence_length), for use with RoPE.

        Returns:
            Output tensor with shape (..., sequence_length, d_model).
        """
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(0,seq_len,1)
        if self.rope is None:
            self.rope = lambda x,y : x
        x_proj = torch.einsum('...d,o d -> ...o',x, self.W).reshape(shape=x.shape[:-2]+(seq_len,self.num_heads,3*self.d_k)).transpose(-3,-2)
        causal_mask = torch.tril(torch.ones(size=(seq_len,seq_len),device=x.device)).bool()
        multi_head_attn = scaled_dot_product_attention(self.rope(x_proj[...,:self.d_k],token_positions.unsqueeze(-2)),self.rope(x_proj[...,self.d_k:2*self.d_k],token_positions.unsqueeze(-2)),x_proj[...,2*self.d_k:3*self.d_k],causal_mask).transpose(-3,-2).reshape(shape=x.shape)
        return torch.einsum('...s d, o d -> ... s o',multi_head_attn, self.W_O)







