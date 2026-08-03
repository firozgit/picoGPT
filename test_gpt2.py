import numpy as np

import gpt2
import gpt2_v1_batching

rng = np.random.default_rng(0)
n_vocab, n_ctx, n_embd, n_head = 17, 8, 12, 3


def affine(input_size, output_size):
    return {
        "w": rng.standard_normal((input_size, output_size)),
        "b": rng.standard_normal(output_size),
    }


params = {
    "wte": rng.standard_normal((n_vocab, n_embd)),
    "wpe": rng.standard_normal((n_ctx, n_embd)),
    "blocks": [
        {
            "attn": {
                "c_attn": affine(n_embd, 3 * n_embd),
                "c_proj": affine(n_embd, n_embd),
            },
            "mlp": {
                "c_fc": affine(n_embd, 4 * n_embd),
                "c_proj": affine(4 * n_embd, n_embd),
            },
            "ln_1": {
                "g": rng.standard_normal(n_embd),
                "b": rng.standard_normal(n_embd),
            },
            "ln_2": {
                "g": rng.standard_normal(n_embd),
                "b": rng.standard_normal(n_embd),
            },
        }
    ],
    "ln_f": {
        "g": rng.standard_normal(n_embd),
        "b": rng.standard_normal(n_embd),
    },
}
prompt = np.array([1, 4, 7, 2])


def test_gpt2_vs_batched_logits():
    expected = gpt2.gpt2(prompt, **params, n_head=n_head)
    actual = gpt2_v1_batching.gpt2(
        prompt[None, :], **params, n_head=n_head
    )[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
