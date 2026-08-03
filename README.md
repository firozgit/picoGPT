# Extras in this fork

Accompanying blog post: [Batching picoGPT](https://firozshaik.com/writing/inference/01-batching/index.html)

> For the original picoGPT project and its documentation, see [PicoGPT](#picogpt) below.

Two files have been added on top of the upstream picoGPT:

* `gpt2_v1_batching.py` — a **batched / vectorized** rewrite of `gpt2.py`. It
  adds a leading batch axis (`[B, T]`) and vectorizes the attention heads
  (`[B, H, T, d_head]`) so multiple prompts run through the forward pass at
  once, instead of one sequence at a time. Generation is still greedy
  (`argmax`).
* `test_gpt2.py` — a `pytest` that checks the batched implementation produces
  the **same logits** as the original `gpt2.py` on random weights.

## Running the batched model

Single prompt (wrap it in quotes):

```bash
python gpt2_v1_batching.py "Alan Turing theorized that computers would one day become"
```

Multiple prompts in one batch — pass several quoted strings:

```bash
python gpt2_v1_batching.py \
    "Alan Turing theorized that computers would one day become" \
    "The quick brown fox jumps over the" \
    --n_tokens_to_generate 20
```

Flags mirror `gpt2.py`: `--n_tokens_to_generate`, `--model_size`
(`["124M", "355M", "774M", "1558M"]`), and `--models_dir`.

> **Note:** prompts in a single batch must currently tokenize to the **same
> length**, because they are stacked into one `np.array([...])` with no
> padding. Mixed-length batching (left-padding + attention mask) is a natural
> next step.

## Running the tests

```bash
pip install pytest          # already included in requirements.txt
python -m pytest test_gpt2.py -v
```

The test builds a tiny random model (`n_vocab=17, n_embd=12, n_head=3`) and
asserts `gpt2_v1_batching.gpt2(...)` matches `gpt2.gpt2(...)` to within
`rtol=atol=1e-12`, so the batching refactor is verified to be numerically
equivalent to the reference implementation.

# PicoGPT
Accompanying blog post: [GPT in 60 Lines of Numpy](https://jaykmody.com/blog/gpt-from-scratch/)

You've seen [openai/gpt-2](https://github.com/openai/gpt-2).

You've seen [karpathy/minGPT](https://github.com/karpathy/mingpt).

You've even seen [karpathy/nanoGPT](https://github.com/karpathy/nanogpt)!

But have you seen [picoGPT](https://github.com/jaymody/picoGPT)??!?

`picoGPT` is an unnecessarily tiny and minimal implementation of [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) in plain [NumPy](https://numpy.org). The entire forward pass code is [40 lines of code](https://github.com/jaymody/picoGPT/blob/main/gpt2_pico.py#L3-L41).

picoGPT features:
* Fast? ❌ Nah, picoGPT is megaSLOW 🐌
* Training code? ❌ Error, 4️⃣0️⃣4️⃣ not found
* Batch inference? ❌ picoGPT is civilized, single file line, one at a time only
* top-p sampling? ❌ top-k? ❌ temperature? ❌ categorical sampling?! ❌ greedy? ✅
* Readable? `gpt2.py` ✅ `gpt2_pico.py` ❌
* Smol??? ✅✅✅✅✅✅ YESS!!! TEENIE TINY in fact 🤏

A quick breakdown of each of the files:

* `encoder.py` contains the code for OpenAI's BPE Tokenizer, taken straight from their [gpt-2 repo](https://github.com/openai/gpt-2/blob/master/src/encoder.py).
* `utils.py` contains the code to download and load the GPT-2 model weights, tokenizer, and hyper-parameters.
* `gpt2.py` contains the actual GPT model and generation code which we can run as a python script.
* `gpt2_pico.py` is the same as `gpt2.py`, but in even fewer lines of code. Why? Because why not 😎👍.

#### Dependencies
```bash
pip install -r requirements.txt
```
Tested on `Python 3.9.10`.

#### Usage
```bash
python gpt2.py "Alan Turing theorized that computers would one day become"
```

Which generates

```
 the most powerful machines on the planet.

The computer is a machine that can perform complex calculations, and it can perform these calculations in a way that is very similar to the human brain.
```

You can also control the number of tokens to generate, the model size (one of `["124M", "355M", "774M", "1558M"]`), and the directory to save the models:

```bash
python gpt2.py \
    "Alan Turing theorized that computers would one day become" \
    --n_tokens_to_generate 40 \
    --model_size "124M" \
    --models_dir "models"
```
