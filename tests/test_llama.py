import torch
from llama_jaxtyping.llama import LLaMaConfig, LLaMA


@torch.no_grad()
def test_forward():
    """test llama."""

    block_size = 32
    vocab_size = 100
    n_layer = 2
    n_head = 4
    n_embd = 16

    llama_config = LLaMaConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
    )

    batch_size = 3
    seq = 10

    token_sample = torch.randint(
        0, vocab_size, size=(batch_size, seq), dtype=torch.int64
    )

    llama_model = LLaMA(llama_config)
    llama_model.apply(llama_model._init_weights)

    out = llama_model(token_sample)

    assert out.shape == (batch_size, seq, llama_config.padded_vocab_size)
