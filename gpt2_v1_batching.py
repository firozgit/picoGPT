import numpy as np
import fire
from utils import load_encoder_hparams_and_params
import tqdm


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(x, g, b, eps: float = 1e-5):  # reduces over last axis, batch-safe
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(var + eps)
    return g * x + b


def linear(x, w, b):  # [..., j] @ [j, k] + [k] -> [..., k], broadcasts over any leading dims
    return x @ w + b


def softmax(x):  # softmax over the last axis
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def attention(q, k, v, mask):
    # q, k, v: [B, H, T, d_head], mask: [T, T] (broadcasts over B, H)
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1]) + mask # [B, H, T, d_head] @ [B, H, d_head, T] + [T, T] -> [B, H, T, T]
    return softmax(scores) @ v  # [B, H, T, T] @ [B, H, T, d_head] -> [B, H, T, d_head]


def mha(x, c_attn, c_proj, n_head):
    # x: [B, T, n_embd]
    B, T, n_embd = x.shape
    d_head = n_embd // n_head

    # qkv projection: [B, T, n_embd] -> [B, T, 3 * n_embd]
    x = linear(x, **c_attn)

    # split into q, k, v: each [B, T, n_embd]
    q, k, v = np.split(x, 3, axis=-1)

    # split the feature axis into heads and move heads next to batch:
    # [B, T, n_embd] -> [B, T, H, d_head] -> [B, H, T, d_head]
    def split_heads(t):
        return t.reshape(B, T, n_head, d_head).transpose(0, 2, 1, 3)

    q, k, v = split_heads(q), split_heads(k), split_heads(v)

    # causal mask: 0 where j <= i, -1e10 where j > i. [T, T]
    mask = (1 - np.tri(T, dtype=x.dtype)) * -1e10

    # one batched call handles every (batch, head): [B, H, T, d_head]
    out = attention(q, k, v, mask)

    # merge heads back: [B, H, T, d_head] -> [B, T, H, d_head] -> [B, T, n_embd]
    x = out.transpose(0, 2, 1, 3).reshape(B, T, n_embd)

    # output projection: [B, T, n_embd] -> [B, T, n_embd]
    x = linear(x, **c_proj)

    return x


def ffn(x, c_fc, c_proj):
    # project up + activation: [B, T, n_embd] -> [B, T, 4 * n_embd]
    x = gelu(linear(x, **c_fc))
    # project down: [B, T, 4 * n_embd] -> [B, T, n_embd]
    x = linear(x, **c_proj)

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    # pre-norm multi-head attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)
    # pre-norm feed forward
    x = x + ffn(layer_norm(x, **ln_2), **mlp)

    return x


def gpt2(input_ids, wte, wpe, blocks, ln_f, n_head):
    # input_ids: [B, T]
    T = input_ids.shape[-1]

    # token embedding [B, T, n_embd] + positional embedding [T, n_embd] (broadcasts over B)
    x = wte[input_ids] + wpe[np.arange(T)]

    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)  # [B, T, n_embd] -> [B, T, n_embd]

    x = layer_norm(x, **ln_f)  # [B, T, n_embd]

    return x @ wte.T  # [B, T, n_embd] @ [n_embd, n_vocab] -> [B, T, n_vocab]


def generate(input_ids, params, n_head, n_tokens_to_generate):
    # input_ids: [B, T] (int array)
    for _ in tqdm.tqdm(range(n_tokens_to_generate), "generating tokens"):
        logits = gpt2(input_ids, **params, n_head=n_head)  # [B, T, n_vocab]
        next_ids = np.argmax(logits[:, -1], axis=-1)        # greedy: [B]
        input_ids = np.concatenate([input_ids, next_ids[:, None]], axis=-1)

    return input_ids[:, -n_tokens_to_generate:]  # [B, n_tokens_to_generate]


"""
{'n_ctx': 1024, 'n_embd': 768, 'n_head': 12, 'n_layer': 12, 'n_vocab': 50257}
"""
"""
{'blocks': [{'attn': {'c_attn': {'b': [2304], 'w': [768, 2304]},
                      'c_proj': {'b': [768], 'w': [768, 768]}},
             'ln_1': {'b': [768], 'g': [768]},
             'ln_2': {'b': [768], 'g': [768]},
             'mlp': {'c_fc': {'b': [3072], 'w': [768, 3072]},
                     'c_proj': {'b': [768], 'w': [3072, 768]}}},
            ...............more layers],
 'ln_f': {'b': [768], 'g': [768]},
 'wpe': [1024, 768],
 'wte': [50257, 768]}
"""

def main(*prompt, n_tokens_to_generate: int = 10, model_size: str = "124M", models_dir: str = "models"):
    # load encoder, hparams and parameters
    encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")

    # pprint(hparams)
    # pprint(params_shape(params))

    # encode and add a batch axis [T] -> [B, T]
    input_ids = np.array([encoder.encode(t) for t in prompt])

    # generate: [B, T] -> [B, n_tokens_to_generate]
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    outputs = [encoder.decode(row.tolist()) for row in output_ids]
    for prompt, out in zip(prompt, outputs):
        print(f"[prompt] {prompt!r}\n[cont]  {out!r}\n")

    return outputs


if __name__ == "__main__":
    fire.Fire(main)
