import math
from dataclasses import dataclass
import gc
import torch
from einops import rearrange
from torch import Tensor, nn
import copy
from ..math import attention, rope,apply_rope
import torch.nn.functional as F
from typing import Union, Tuple  # Import Tuple and Union from typing
import numpy as np
import math

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def scaled_dot_product_attention2(query, key, value, image_size, dropout_p=0.0, is_causal=False,
                                  attn_mask=None, union_mask=None, local_mask_list=[],
                                  local_t2i_strength=1,
                                  context_t2i_strength=1,
                                  locali2i_strength=1,
                                  local2out_i2i_strength=1,
                                  num_edit_region=1,
                                  scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:

            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attn_mask + attn_bias


    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    if attn_mask.dtype == torch.bool:
        attn_weight += attn_bias
    else:
        attn_weight += attn_bias

    ################# Attention Amplification #################
    # amplify the attention between the local text prompt and local edit region
    if union_mask!=None:
        curr_atten = copy.deepcopy(attn_weight[:, :, -image_size:, 512:512*(num_edit_region+1)])
        attn_weight[:, :, -image_size:, 512:512*(num_edit_region+1)] =torch.where(union_mask == 1, curr_atten,curr_atten * (local_t2i_strength)) ##attn_weight[:, :, -image_size:, 512:512*(num_edit_region+1)]*union_mask #
        # amplify the attention between the target prompt and the whole image
        curr_atten1 = copy.deepcopy(attn_weight[:, :, -image_size:, :512])
        attn_weight[:, :, -image_size:, :512] = curr_atten1 * (context_t2i_strength)

        #0811 what we do should be: each edit region see itself = locali2i_strength, each edit region see outside=local2out_i2i_strength
        #currently it is doing:each edit region see itself AND OTHER EDIT REGIONS = locali2i_strength
        mask1_flat = (union_mask).flatten()
        mask2_flat = (1 - union_mask).flatten()
        mask2_indices = 512 * (num_edit_region+1) + torch.nonzero(mask2_flat, as_tuple=True)[0]
        mask1_indices = 512 * (num_edit_region+1) + torch.nonzero(mask1_flat, as_tuple=True)[0]
        attn_weight[:, :, mask2_indices[:, None], mask1_indices] = local2out_i2i_strength * attn_weight[:, :, mask2_indices[:, None],mask1_indices]
        attn_weight[:, :, mask2_indices[:, None], mask2_indices] =  locali2i_strength * attn_weight[:, :, mask2_indices[:, None],mask2_indices]
        for local_mask in local_mask_list:
            ## outside the union of masks is 1
            mask1_flat = union_mask.flatten()#(local_mask).flatten()
            mask1_indices = 512 * (num_edit_region + 1) + torch.nonzero(mask1_flat, as_tuple=True)[0]
            ## mask2_flat inside the mask is 1
            mask2_flat = (1 - local_mask).flatten()
            mask2_indices = 512 * (num_edit_region + 1) + torch.nonzero(mask2_flat, as_tuple=True)[0]
            ## inside the other masks is 1
            mask3_flat = 1-torch.logical_or(mask1_flat.bool(), mask2_flat.bool()).int()
            mask3_indices = 512 * (num_edit_region + 1) + torch.nonzero(mask3_flat, as_tuple=True)[0]

            # amplify the attention within the edit region
            attn_weight[:, :, mask2_indices[:, None], mask2_indices] = locali2i_strength * attn_weight[:, :,mask2_indices[:, None],mask2_indices]
            # amplify the attention between the edit region and the bg region
            attn_weight[:, :, mask2_indices[:, None], mask1_indices] = local2out_i2i_strength * attn_weight[:, :, mask2_indices[:, None],mask1_indices]
            # amplify the attention between the edit region and other edit regions
            attn_weight[:, :, mask2_indices[:, None], mask3_indices] = local2out_i2i_strength * attn_weight[:, :,mask2_indices[:, None],mask3_indices]


    ##############END of Amplification###########################

    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    return attn_weight @ value






class EmbedND(nn.Module):
    def __init__(self, dim: int, theta: int, axes_dim: list[int]):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: Tensor) -> Tensor:
        n_axes = ids.shape[-1]
        emb = torch.cat(
            [rope(ids[..., i], self.axes_dim[i], self.theta) for i in range(n_axes)],
            dim=-3,
        )

        return emb.unsqueeze(1)


def timestep_embedding(t: Tensor, dim, max_period=10000, time_factor: float = 1000.0):
    """
    Create sinusoidal timestep embeddings.
    :param t: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an (N, D) Tensor of positional embeddings.
    """
    t = time_factor * t
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
        t.device
    )

    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    if torch.is_floating_point(t):
        embedding = embedding.to(t)
    return embedding


class MLPEmbedder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.in_layer = nn.Linear(in_dim, hidden_dim, bias=True)
        self.silu = nn.SiLU()
        self.out_layer = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.out_layer(self.silu(self.in_layer(x)))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        x_dtype = x.dtype
        x = x.float()
        rrms = torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + 1e-6)
        return (x * rrms).to(dtype=x_dtype) * self.scale


class QKNorm(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.query_norm = RMSNorm(dim)
        self.key_norm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)

class FLuxSelfAttnProcessor:
    def __call__(self, attn, x, pe, **attention_kwargs):


        qkv = attn.qkv(x)
        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=self.num_heads)
        q, k = attn.norm(q, k, v)
        x = attention(q, k, v, pe=pe)
        x = attn.proj(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm = QKNorm(head_dim)
        self.proj = nn.Linear(dim, dim)
    def forward():
        pass


@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor


class Modulation(nn.Module):
    def __init__(self, dim: int, double: bool):
        super().__init__()
        self.is_double = double
        self.multiplier = 6 if double else 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor) -> Tuple[ModulationOut, Union[ModulationOut, None]]:
    #def forward(self, vec: Tensor) -> tuple[ModulationOut, ModulationOut | None]:
        out = self.lin(nn.functional.silu(vec))[:, None, :].chunk(self.multiplier, dim=-1)

        return (
            ModulationOut(*out[:3]),
            ModulationOut(*out[3:]) if self.is_double else None,
        )


class DoubleStreamBlockProcessor:
    def __call__(self, attn, img, txt, vec, pe, attention_kwargs):

        img_mod1, img_mod2 = attn.img_mod(vec)
        txt_mod1, txt_mod2 = attn.txt_mod(vec)

        # prepare image for attention
        img_modulated = attn.img_norm1(img)
        img_modulated = (1 + img_mod1.scale) * img_modulated + img_mod1.shift
        img_qkv = attn.img_attn.qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)
        img_q, img_k = attn.img_attn.norm(img_q, img_k, img_v)


        # prepare txt for attention
        txt_modulated = attn.txt_norm1(txt)
        txt_modulated = (1 + txt_mod1.scale) * txt_modulated + txt_mod1.shift

        txt_qkv = attn.txt_attn.qkv(txt_modulated)

        txt_q, txt_k, txt_v = rearrange(txt_qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads, D=attn.head_dim)

        txt_q, txt_k = attn.txt_attn.norm(txt_q, txt_k, txt_v)

        output_indicator=0

        if 'regional_attention_mask' in attention_kwargs:
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)
            q, k = apply_rope(q, k, pe)
            attention_mask = attention_kwargs['regional_attention_mask']
            if 'union_mask' in attention_kwargs:


                x = scaled_dot_product_attention2(q, k, v, attention_kwargs['image_size'],
                                                               dropout_p=0.0, is_causal=False,
                                                               attn_mask=attention_mask,
                                                               union_mask=attention_kwargs['union_mask'],
                                                               local_mask_list=attention_kwargs[
                                                                   'local_mask_all_dilate'],
                                                               local_t2i_strength=attention_kwargs[
                                                                   'local_t2i_strength'],
                                                               context_t2i_strength=attention_kwargs[
                                                                   'context_t2i_strength'],
                                                               locali2i_strength=attention_kwargs[
                                                                   'local_i2i_strength'],
                                                               local2out_i2i_strength=attention_kwargs[
                                                                   'local2out_i2i_strength'],
                                                               num_edit_region=attention_kwargs['num_edit_region'])


            else:
                x = scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False, attn_mask=attention_mask)


            attn1 = rearrange(x, "B H L D -> B L (H D)")
            txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1]:]
        else:
            q = torch.cat((txt_q, img_q), dim=2)
            k = torch.cat((txt_k, img_k), dim=2)
            v = torch.cat((txt_v, img_v), dim=2)
            attn1 = attention(q, k, v, pe=pe)
            txt_attn, img_attn = attn1[:, : txt.shape[1]], attn1[:, txt.shape[1]:]


        # calculate the img bloks
        img = img + img_mod1.gate * attn.img_attn.proj(img_attn)
        img = img + img_mod2.gate * attn.img_mlp((1 + img_mod2.scale) * attn.img_norm2(img) + img_mod2.shift)

        # calculate the txt bloks
        txt = txt + txt_mod1.gate * attn.txt_attn.proj(txt_attn)
        txt = txt + txt_mod2.gate * attn.txt_mlp((1 + txt_mod2.scale) * attn.txt_norm2(txt) + txt_mod2.shift)


        return img, txt




class DoubleStreamBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float, qkv_bias: bool = False):
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        self.img_mod = Modulation(hidden_size, double=True)
        self.img_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.img_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.img_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        self.txt_mod = Modulation(hidden_size, double=True)
        self.txt_norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_attn = SelfAttention(dim=hidden_size, num_heads=num_heads, qkv_bias=qkv_bias)

        self.txt_norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.txt_mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )
        processor = DoubleStreamBlockProcessor()
        self.set_processor(processor)

    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        image_proj: Tensor = None,
        ip_scale: float =1.0,
        attention_kwargs={}
    ) -> tuple[Tensor, Tensor]:

        if image_proj is None:
            return self.processor(self, img, txt, vec, pe,attention_kwargs)
        else:
            return self.processor(self, img, txt, vec, pe, image_proj, ip_scale)




class SingleStreamBlockProcessor:
    def __call__(self, attn: nn.Module, x: Tensor, vec: Tensor, pe: Tensor,attention_kwargs) -> Tensor:


        mod, _ = attn.modulation(vec)
        x_mod = (1 + mod.scale) * attn.pre_norm(x) + mod.shift
        qkv, mlp = torch.split(attn.linear1(x_mod), [3 * attn.hidden_size, attn.mlp_hidden_dim], dim=-1)

        q, k, v = rearrange(qkv, "B L (K H D) -> K B H L D", K=3, H=attn.num_heads)
        q, k = attn.norm(q, k, v)

        ## change
        if 'regional_attention_mask' in attention_kwargs:

            q, k = apply_rope(q, k, pe)
            attention_mask = attention_kwargs['regional_attention_mask']

            if  'union_mask' in attention_kwargs:


                attn_1 = scaled_dot_product_attention2(q, k, v, attention_kwargs['image_size'],
                                                                    dropout_p=0.0,
                                                                    is_causal=False,
                                                                    attn_mask=attention_mask,
                                                                    union_mask=attention_kwargs['union_mask'],
                                                                    local_mask_list=attention_kwargs['local_mask_all_dilate'],
                                                                    local_t2i_strength=attention_kwargs['local_t2i_strength'],
                                                                    context_t2i_strength=attention_kwargs['context_t2i_strength'],
                                                                    locali2i_strength=attention_kwargs['local_i2i_strength'],
                                                                    local2out_i2i_strength=attention_kwargs['local2out_i2i_strength'],
                                                                    num_edit_region=attention_kwargs['num_edit_region'])

            else:
                attn_1 = scaled_dot_product_attention(q, k, v, dropout_p=0.0,
                                                                     is_causal=False, attn_mask=attention_mask)


            attn_1 = rearrange(attn_1, "B H L D -> B L (H D)")
        else:
            attn_1 = attention(q, k, v, pe=pe)

        # compute activation in mlp stream, cat again and run second linear layer
        output = attn.linear2(torch.cat((attn_1, attn.mlp_act(mlp)), 2))
        output = x + mod.gate * output


        return output


class SingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        #qk_scale: float | None = None,
        qk_scale: Union[float, None] = None

    ):
        super().__init__()
        self.hidden_dim = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.mlp_hidden_dim = int(hidden_size * mlp_ratio)
        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + self.mlp_hidden_dim)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + self.mlp_hidden_dim, hidden_size)

        self.norm = QKNorm(self.head_dim)

        self.hidden_size = hidden_size
        self.pre_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        self.mlp_act = nn.GELU(approximate="tanh")
        self.modulation = Modulation(hidden_size, double=False)

        processor = SingleStreamBlockProcessor()
        self.set_processor(processor)


    def set_processor(self, processor) -> None:
        self.processor = processor

    def get_processor(self):
        return self.processor

    def forward(
        self,
        x: Tensor,
        vec: Tensor,
        pe: Tensor,
        #image_proj: Tensor | None = None,
        image_proj: Union[Tensor, None] = None,
        ip_scale: float = 1.0,
        attention_kwargs={}
    ) -> Tensor:
        if image_proj is None:
            return self.processor(self, x, vec, pe,attention_kwargs)
        else:
            return self.processor(self, x, vec, pe, image_proj, ip_scale)



class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x: Tensor, vec: Tensor) -> Tensor:
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        x = (1 + scale[:, None, :]) * self.norm_final(x) + shift[:, None, :]
        x = self.linear(x)
        return x
