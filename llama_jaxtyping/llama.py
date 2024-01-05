# credits to https://github.com/Lightning-AI/lit-llama/blob/main/lit_llama/model.py
# for the original implementation of the llama model

"""Full definition of a LLaMA Language Model, all of it in this single file.

Based on the nanoGPT implementation: https://github.com/karpathy/nanoGPT.
"""
# mypy: ignore-errors
import math
from typing import Any, TypeAlias

from pydantic import BaseModel, model_validator
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import Tensor
from jaxtyping import Float, Int, jaxtyped
from beartype import beartype as typechecker

MaskCache = torch.Tensor
RoPECache = Float[Tensor, "block_size head_embd 2"]
KVCache = tuple[torch.Tensor, torch.Tensor]  # todo use jaxtyping


HiddenState: TypeAlias = Float[Tensor, "batch seq n_embd"]
QKV: TypeAlias = Float[Tensor, "batch seq n_head head_embd"]
Logits: TypeAlias = Float[Tensor, "batch seq vocab_size"]
Index: TypeAlias = Int[Tensor, "batch seq"]


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class LLaMaConfig(BaseModel):
    block_size: int
    vocab_size: int
    padded_vocab_size: int
    n_layer: int
    n_head: int
    n_embd: int

    @model_validator(mode="before")
    def set_padded_vocab_size(cls, values: dict[str, Any]):
        """Set the padded vocab size to the next multiple of 64 if not provided."""
        vocab_size = values.get("vocab_size")
        padded_vocab_size = values.get("padded_vocab_size")

        if padded_vocab_size is None and vocab_size is not None:
            values["padded_vocab_size"] = find_multiple(vocab_size, 64)
        return values


class LLaMA(nn.Module):
    def __init__(self, conf: LLaMaConfig) -> None:
        super().__init__()
        assert conf.padded_vocab_size is not None
        self.conf = conf

        self.lm_head = nn.Linear(conf.n_embd, conf.padded_vocab_size, bias=False)
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(conf.padded_vocab_size, conf.n_embd),
                h=nn.ModuleList(Block(conf) for _ in range(conf.n_layer)),
                ln_f=RMSNorm(conf.n_embd),
            )
        )

        self.rope_cache: RoPECache | None = None
        self.mask_cache: MaskCache | None = None
        self.kv_caches: list[KVCache] = []

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.conf.n_layer)
            )
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(
                module.weight, mean=0.0, std=0.02 / math.sqrt(2 * self.conf.n_layer)
            )

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        idx: Index,
        max_seq_length: int | None = None,
        input_pos: torch.Tensor | None = None,
    ) -> Logits | tuple[Logits, list[KVCache]]:
        B, T = idx.size()

        block_size = self.conf.block_size
        if max_seq_length is None:
            max_seq_length = block_size
        assert (
            T <= max_seq_length
        ), f"Cannot forward sequence of length {T}, max seq length is only {max_seq_length}"
        assert (
            max_seq_length <= block_size
        ), f"Cannot attend to {max_seq_length}, block size is only {block_size}"
        assert (
            T <= block_size
        ), f"Cannot forward sequence of length {T}, block size is only {block_size}"

        if self.rope_cache is None:
            self.rope_cache = self.build_rope_cache(idx)
        if self.mask_cache is None:
            self.mask_cache = self.build_mask_cache(idx)

        if input_pos is not None:
            rope = self.rope_cache.index_select(0, input_pos)
            mask = self.mask_cache.index_select(2, input_pos)
            mask = mask[:, :, :, :max_seq_length]
        else:
            rope = self.rope_cache[:T]
            mask = self.mask_cache[:, :, :T, :T]

        # forward the model itself
        x = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        if input_pos is None:  # proxy for use_cache=False
            for block in self.transformer.h:
                x, _ = block(x, rope, mask, max_seq_length)
        else:
            if not self.kv_caches:
                head_size = self.conf.n_embd // self.conf.n_head
                cache_shape = (B, self.conf.n_head, max_seq_length, head_size)
                self.kv_caches = [
                    (
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                        torch.zeros(cache_shape, device=x.device, dtype=x.dtype),
                    )
                    for _ in range(self.conf.n_layer)
                ]
            for i, block in enumerate(self.transformer.h):
                x, self.kv_caches[i] = block(
                    x, rope, mask, max_seq_length, input_pos, self.kv_caches[i]
                )

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)  # (b, t, vocab_size)

        return logits

    @jaxtyped(typechecker=typechecker)
    def build_rope_cache(self, idx: Index) -> RoPECache:
        return build_rope_cache(
            seq_len=self.conf.block_size,
            n_elem=self.conf.n_embd // self.conf.n_head,
            dtype=idx.dtype,
            device=idx.device,
        )

    @jaxtyped(typechecker=typechecker)
    def build_mask_cache(self, idx: Index) -> MaskCache:
        ones = torch.ones(
            (self.conf.block_size, self.conf.block_size),
            device=idx.device,
            dtype=torch.bool,
        )
        return torch.tril(ones).unsqueeze(0).unsqueeze(0)

    def reset_cache(self) -> None:
        self.kv_caches.clear()
        if self.mask_cache.device.type == "xla":
            # https://github.com/Lightning-AI/lit-parrot/pull/83#issuecomment-1558150179
            self.rope_cache = None
            self.mask_cache = None


class Block(nn.Module):
    def __init__(self, conf: LLaMaConfig) -> None:
        super().__init__()
        self.rms_1 = RMSNorm(conf.n_embd)
        self.attn = CausalSelfAttention(conf)
        self.rms_2 = RMSNorm(conf.n_embd)
        self.mlp = MLP(conf)

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: HiddenState,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[HiddenState, KVCache | None]:
        h, new_kv_cache = self.attn(
            self.rms_1(x), rope, mask, max_seq_length, input_pos, kv_cache
        )
        x = x + h
        x = x + self.mlp(self.rms_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, conf: LLaMaConfig) -> None:
        super().__init__()
        assert conf.n_embd % conf.n_head == 0

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(conf.n_embd, 3 * conf.n_embd, bias=False)
        # output projection
        self.c_proj = nn.Linear(conf.n_embd, conf.n_embd, bias=False)

        self.n_head = conf.n_head
        self.n_embd = conf.n_embd
        self.block_size = conf.block_size

    @jaxtyped(typechecker=typechecker)
    def forward(
        self,
        x: HiddenState,
        rope: RoPECache,
        mask: MaskCache,
        max_seq_length: int,
        input_pos: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
    ) -> tuple[HiddenState, KVCache | None]:
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        head_size = C // self.n_head
        k = k.view(B, T, self.n_head, head_size)
        q = q.view(B, T, self.n_head, head_size)
        v = v.view(B, T, self.n_head, head_size)

        q = apply_rope(q, rope)
        k = apply_rope(k, rope)

        k = k.transpose(1, 2)  # (B, nh, T, hs)
        q = q.transpose(1, 2)  # (B, nh, T, hs)
        v = v.transpose(1, 2)  # (B, nh, T, hs)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            # check if reached token limit
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                # shift 1 position to the left
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy(2, input_pos, k)
            v = cache_v.index_copy(2, input_pos, v)
            kv_cache = k, v

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        #  att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #  att = att.masked_fill(mask[:,:,:T,:T] == 0, float('-inf'))
        #  att = F.softmax(att, dim=-1)
        #  y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # efficient attention using Flash Attention CUDA kernels
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.c_proj(y)

        return y, kv_cache


class MLP(nn.Module):
    def __init__(self, conf: LLaMaConfig) -> None:
        super().__init__()
        hidden_dim = 4 * conf.n_embd
        n_hidden = int(2 * hidden_dim / 3)
        n_hidden = find_multiple(n_hidden, 256)

        self.c_fc1 = nn.Linear(conf.n_embd, n_hidden, bias=False)
        self.c_fc2 = nn.Linear(conf.n_embd, n_hidden, bias=False)
        self.c_proj = nn.Linear(n_hidden, conf.n_embd, bias=False)

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: HiddenState) -> HiddenState:
        x = F.silu(self.c_fc1(x)) * self.c_fc2(x)
        x = self.c_proj(x)
        return x


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Derived from https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py. BSD 3-Clause License:
    https://github.com/bzhangGo/rmsnorm/blob/master/LICENSE.
    """

    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    @jaxtyped(typechecker=typechecker)
    def forward(self, x: HiddenState) -> HiddenState:
        # NOTE: the original RMSNorm paper implementation is not equivalent
        # norm_x = x.norm(2, dim=self.dim, keepdim=True)
        # rms_x = norm_x * d_x ** (-1. / 2)
        # x_normed = x / (rms_x + self.eps)
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.scale * x_normed


@jaxtyped(typechecker=typechecker)
def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
) -> RoPECache:
    """Enhanced Transformer with Rotary Position Embedding.

    Derived from: https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/
    transformers/rope/__init__.py. MIT License:
    https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/license.
    """
    # $\Theta = {\theta_i = 10000^{\frac{2(i-1)}{d}}, i \in [1, 2, ..., \frac{d}{2}]}$
    theta = 1.0 / (
        base ** (torch.arange(0, n_elem, 2, dtype=dtype, device=device) / n_elem)
    )

    # Create position indexes `[0, 1, ..., seq_len - 1]`
    seq_idx = torch.arange(seq_len, dtype=dtype, device=device)

    # Calculate the product of position index and $\theta_i$
    idx_theta = torch.outer(seq_idx, theta).float()

    cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)

    # this is to mimic the behaviour of complex32, else we will get different results``
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        cache = cache.half()
    return cache


def apply_rope(x: QKV, rope_cache: RoPECache) -> QKV:
    # truncate to support variable sizes
    T = x.size(1)
    rope_cache = rope_cache[:T]

    # cast because the reference does
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    rope_cache = rope_cache.view(1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)
