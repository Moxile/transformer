# AI Papers — Study Track

**Goal:** enough depth to hold a real conversation with people who build frontier models, and enough implementation skill to test my own ideas without being blocked by what's already known.

**Method:** projects are the spine. Papers get read *because a project needs them*, not in order. Reading a paper before you have a question it answers is how you forget it.

**Depth: 3 = implement it · 2 = read closely · 1 = skim for the finding.**

A depth-3 paper is only done when something runs. If nothing runs, it was a 2.

---

## Phase 0 — Foundations (implement these first, ~2 weeks)

No project context needed. This is the vocabulary everything else assumes.

| # | Paper | Depth | Artifact |
|---|---|---|---|
| 01 | Attention Is All You Need | 3 | transformer from scratch, trained on something small |
| 02 | RoPE | 3 | RoPE + one scaling variant, correctness-tested |
| 17 | ViT | 3 | ViT trained on a real dataset |
| 11 | Mixed Precision Training | 3 | bf16 loop; deliberately induce a NaN, then fix it |
| 03 | GPT-2 / BERT | 1 | — |
| 04 | Chinchilla | 1 | — |

---

## Phase 1 — Project A: text-to-image from scratch

**Build:** your own tokenizer → latent diffusion → transformer backbone → flow matching. Small scale, single GPU, end to end. This is the highest-value project on the list: it touches nine papers, and every architectural call in it is a real one.

Read each paper the week you need it, in this order.

| # | Paper | Depth | What it unblocks |
|---|---|---|---|
| 19 | VAE | 2 | the latent space you'll diffuse in |
| 20 | VQ-VAE | 3 | your tokenizer; measure codebook usage |
| 21 | VQGAN | 2 | why the VAE needs perceptual + adversarial loss |
| 22 | DDPM | 3 | first working sampler, on CIFAR-10 |
| 23 | Classifier-Free Guidance | 3 | conditioning + a guidance-scale sweep |
| 18 | CLIP | 2 | text conditioning |
| 24 | Latent Diffusion | 3 | move diffusion into your latents |
| 25 | DiT | 3 | transformer backbone; **ablate adaLN-Zero** |
| 26 | Flow Matching | 3 | swap the objective, compare against 22 |
| 27 | SD3 | 2 | MMDiT — how it's done at production scale |
| 28 | Movie Gen | 1 | — |

**The ablation is the point.** Anyone can train a DiT from a repo. Changing the modulation scheme and being able to say what happened and why is the thing that makes you a peer.

---

## Phase 2 — Project B: a small LLM, pretrain through post-training

**Build:** a ~100M-parameter model, pretrained on a real corpus, then SFT'd and preference-tuned until it holds a conversation. Badly, but genuinely — and you'll have built every stage yourself.

This is the ChatGPT lineage end to end. It's what makes the conversation you want possible, because ChatGPT is mostly a post-training story and that's the part outsiders never understand.

| # | Paper | Depth | What it unblocks |
|---|---|---|---|
| 10 | FlashAttention | 3 | Triton fused attention kernel |
| 08 | Mixture of Experts | 2 | routing as an architectural option |
| 09 | InstructGPT | 2 | the SFT → RM → PPO pipeline as a whole |
| 29 | DPO | 3 | preference tuning you'll actually run |
| 06 | LoRA | 3 | implement by hand; cheap iteration on top |
| 30 | DeepSeek-R1 | 2 | reasoning-model training — the current frontier |

---

## Phase 3 — Scale sprint (rented GPUs, ~1 week)

Everything above runs on one card. This phase is the part self-study almost always skips, and it's the clearest separator in a technical conversation. Rent 8 GPUs for a few days and run Project B's model under each parallelism strategy.

| # | Paper | Depth | Artifact |
|---|---|---|---|
| 12 | ZeRO | 3 | multi-GPU FSDP run, profiled |
| 13 | Megatron-LM | 3 | TP on one block, correctness-checked against single-GPU |
| 14 | GPipe | 2 | — |
| 15 | Activation Recomputation | 2 | — |
| 16 | Llama 3 Herd | 3 | notes: failure → symptom → fix, as a table |

Read 16 *after* your own multi-node run breaks. It reads completely differently once you've seen a throughput regression yourself.

---

## Cut

Learn from code, docs, or by getting burned — not from papers.

| Topic | Instead |
|---|---|
| ResNet | Absorbed by everything above. |
| NNUE (chess) | Stockfish repo, out of scope for this list. |
| QLoRA | `bitsandbytes` docs when you need 4-bit. |
| Inference / serving | vLLM docs + scheduler source. |
| RMSNorm / SwiGLU / GQA | A modern `modeling_*.py` *is* the reference. |
| Tokenization (BPE) | Write one in 100 lines. |
| Data curation, evaluation | Learned on your own runs. Nothing transfers from papers. |
| CUDA / Triton | Triton tutorials, then paper 10. |

---

## Honest scope note

21 papers with 14 implementations is more than one summer. If time runs short, **finish Project A completely** rather than starting Project B. One end-to-end system you can defend in detail beats two half-built ones — in a conversation and on a CV.
