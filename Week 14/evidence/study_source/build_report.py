from pathlib import Path
import json

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parent/'Session_14_MoE_Extended'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
a=read(ROOT/'architecture_numbers.json')
c=a['parameters']
v=read(RUN/'moe/final_validation.json')
d=read(RUN/'dense/final_validation.json')
s=read(RUN/'diagnostics/summary.json')
lines=[r'''**Our trained MoE: complete architecture and expert inventory**

This describes the final checkpoint after 50M continuation targets: 10M in the first experiment and 40M in the extension. The architecture stayed the same throughout. Counts were checked by strictly loading the saved checkpoint and counting its actual tensors. The session PDF is reference material; its Qwen example is not the model we trained.

**The model table, in the style of the PDF's first table (page 3)**

| Part | Our trained model | PDF reference: Qwen3-30B-A3B |
|---|---:|---:|
| Transformer layers | 9 | 48 |
| Layers containing MoE | All 9 | All 48 |
| Hidden size | 384 | 2,048 |
| Query heads | 6 | 32 |
| Key/value heads | 6 | 4 |
| Dimension per attention head | 64 | 128 |
| Attention type | Full multi-head attention | Grouped-query attention |
| Routed experts per layer | 4 | 128 |
| Experts selected per token per layer | 2 | 8 |
| Shared, always-on experts per layer | 0 | 0 |
| Expert intermediate width | 1,664 | 768 |
| Matrices in each expert | 2: up and down, with GELU | 3: gate, up and down |
| Vocabulary | 8,192 | 151,936 |
| Unique expert modules across model | 36 | 6,144 |
| Expert applications on one token through all layers | 18 | 384 |
| Total parameters | 54,685,440 | Approximately 30.53B in the PDF |
| Active parameters per token | 31,682,304 | Approximately 3.35B in the PDF |

The Qwen column reproduces the supplied PDF's example, not an independently audited Qwen checkpoint. Our exact counts use tied input/output embeddings. Expert counts are local to a layer: layer 1 E0 and layer 2 E0 are different networks with different weights.

**Other architecture settings**

| Setting | Actual value |
|---|---|
| Model family | Autoregressive, decoder-only, pre-LayerNorm Transformer |
| Maximum context | 512 tokens |
| Position representation | Learned 512 × 384 table; position indices reset at packed document fragments |
| Token embedding | 8,192 × 384 |
| Output projection | 384 → 8,192, tied to the token embedding; no separate output weight table |
| Normalization | LayerNorm, epsilon 0.00001, learned scale and bias |
| LayerNorm modules | 2 per block × 9 + 1 final = 19 |
| Attention mask | Causal, with no attention across packed document segments |
| Query/key/value projection | Combined 384 → 1,152, then 6 heads × 64 for each of Q, K and V |
| Attention output projection | 384 → 384 |
| Expert activation | GELU, tanh approximation |
| Linear-layer biases | None in attention, experts, router or output projection |
| Dropout | 0 |
| Weight quantization | None |
| Activation checkpointing | Not implemented |
| KV cache | Not implemented; the generation helper recomputes the current context |
| RoPE, GQA, SwiGLU | Not used in our model |
| Router modules | 9 independent trainable routers |
| Expert weight sharing | None between experts or layers after copying |
| Frozen parameters | None |

**What happens inside a layer**

The input is a 384-number representation for every token. Attention first mixes information from earlier tokens in the same document. The FFN/MoE then transforms each resulting contextual token vector independently.

```text
token IDs → token embeddings + learned position embeddings
  → block 1 → block 2 → ... → block 9
  → final LayerNorm → tied vocabulary projection → next-token logits

Each block:
  x → LayerNorm → causal self-attention → output projection → add x
    → LayerNorm → router scores E0, E1, E2, E3
                → choose 2 experts
                → run their FFNs on this token
                → normalized weighted sum → residual addition
```

Each expert computes `down(GELU(up(x)))`. The up projection maps 384 to 1,664 numbers, and the down projection returns 384 numbers. Stored PyTorch tensor shapes are `[1664, 384]` and `[384, 1664]`.

| One expert component | Parameters |
|---|---:|
| Up matrix: 384 × 1,664 | 638,976 |
| Down matrix: 1,664 × 384 | 638,976 |
| Biases | 0 |
| GELU's learned parameters | 0 |
| Total per expert | **1,277,952** |

An expert has no attention, tokenizer, output vocabulary head or internal normalization of its own. It is an FFN, not an independent language model. Its width is 1,664 / 384 = 4.3333 times the hidden size. Four stored experts give an aggregate intermediate width of 6,656; two selected experts give 3,328. The original dense block used one width-1,664 FFN, so this conversion doubles the active expert FFN width relative to that original block.

**Every layer's expert count and parameter budget**

The 'common' count includes attention, the two LayerNorms and the router. It does not mean an always-on shared expert.

| Layer (human numbering) | Routed experts | Selected/token | Shared experts | Router parameters | Total block parameters | Active block parameters |
|---:|---:|---:|---:|---:|---:|---:|''']
for x in a['layers']:
    lines.append(f"| {x['layer']} | {x['experts']} | {x['selected']} | 0 | {x['router_parameters']:,} | {x['total_parameters']:,} | {x['active_parameters']:,} |")
lines += [r'''
The code numbers these layers 0–8. Every layer stores four experts, executes two on each non-padding token, and skips the other two for that token. Across nine layers there are 36 stored experts, 18 selected expert applications per token and 18 unselected experts relative to that token. A batch can use all 36.

**Exact parameter accounting, analogous to the PDF's second table (page 5)**

| Component | Per layer | Across model |
|---|---:|---:|
| QKV projection | 442,368 | 3,981,312 |
| Attention output projection | 147,456 | 1,327,104 |
| Both block LayerNorms, scale + bias | 1,536 | 13,824 |
| Router, 384 × 4 | 1,536 | 13,824 |
| All four experts | 5,111,808 | 46,006,272 |
| Token embedding / tied output weight | — | 3,145,728 |
| Position embedding | — | 196,608 |
| Final LayerNorm, scale + bias | — | 768 |
| **Total** | **5,704,704 per block** | **54,685,440** |

The total outside the nine blocks is 3,343,104. Each block has 592,896 common parameters and 5,111,808 expert parameters. With top-2, 2,555,904 expert parameters are selected and 2,555,904 are unselected per block per token.

```text
Common model parameters = 8,679,168
All expert parameters   = 36 × 1,277,952 = 46,006,272
Total                   = 8,679,168 + 46,006,272 = 54,685,440

Selected expert weights = 9 × 2 × 1,277,952 = 23,003,136
Active per token        = 8,679,168 + 23,003,136 = 31,682,304
Inactive per token      = 54,685,440 − 31,682,304 = 23,003,136
```

Active parameters are 57.94% of the whole model; inactive parameters are 42.06%. Within the expert bank alone, the split is exactly 50%/50%. This architectural convention includes the entire embedding and position tables rather than only lookup rows; the tied token embedding also serves the full vocabulary output projection at predicted positions. It is not a measured FLOP count or a statement that every counted table element is read for every input position.

The original dense model has 20,166,912 parameters. This MoE has 2.7116 times its total parameters and 1.5710 times its active parameters. Sparse routing saves expert work compared with executing all four experts, but our top-2 full-copy conversion does more work than the original one-FFN dense model. These experiments are matched for token exposure, not compute or parameter count.

**Router: exact operation and size**

Each layer's router is one bias-free linear projection from 384 inputs to four scores. Its matrix has 1,536 weights, stored as `[4, 384]`; all nine routers together have 13,824 weights. It has no hidden layer. At conversion, its weights were initialized from a normal distribution with mean 0 and standard deviation 0.01.

For each non-padding token:

1. Apply the router to the normalized 384-dimensional contextual representation.
2. Compute softmax over all four scores in FP32.
3. Select the two largest probabilities, with distinct expert IDs.
4. Divide those two probabilities by their sum so the selected weights sum to one.
5. Gather this token into the corresponding experts' input subsets.
6. Compute each selected expert's output and add the two weighted outputs.

Example probabilities `[0.10, 0.45, 0.35, 0.10]` choose E1 and E2. Their mixture weights are 0.5625 and 0.4375, so the MoE output is `0.5625 × E1(x) + 0.4375 × E2(x)`. This is an illustration, not a recorded token's routing.

There are six possible unordered top-2 pairs per layer. Choices are per token and per layer, not one pair for an entire sentence. The same token ID can take different paths in different contexts. The implementation genuinely dispatches subsets of tokens to each expert; it does not evaluate four FFNs on every token and mask two afterward. The four expert modules are visited in a Python loop on one GPU.

Capacity is not capped: there is no capacity factor, overflow queue or token dropping. Padding tokens are excluded from expert dispatch; context-only non-padding tokens are still routed, even when their positions are not directly scored by the target loss. That is why routing counts and supervised target counts differ.

**How experts and routers learn**

We used full-copy upcycling. The source dense Transformer had already trained on 50M supervised targets. At each layer, we copied that layer's entire trained FFN four times. We also copied the attention, embeddings and normalization parameters. There was no partition of the original FFN into smaller pieces. Only the new routers started from random weights.

At the moment of copying, all four experts in a layer implement the same function. Any normalized mixture of identical outputs reproduces that function, up to floating-point differences. As routing sends different tokens to different copies, their gradients and weights diverge. Initially identical outputs provide little/no task signal for relative mixture weights; learned differences between experts subsequently make those weights matter, while the balancing objective can train the router from the start.

All weights remain trainable: experts, routers, attention, embeddings and normalizations. Selected experts receive the task gradient carried by the tokens routed through them, weighted by the mixture. Attention passes contextual information to the expert; the expert itself applies the same learned transformation to all tokens it receives. The router learns from the task loss through the selected mixture weights and from the auxiliary balancing loss. The discrete top-k choice is not itself differentiated; no straight-through estimator or reinforcement learning is implemented.

For one token, an unselected expert receives no task gradient contribution from that token. It can still receive gradients from other tokens in the batch. Furthermore, optimizer momentum and weight decay can move a parameter with zero current task gradient when that parameter participates in the optimizer step. 'Inactive for this token' does not mean frozen, removed from memory or never trained.

The balancing objective per layer is `4 × sum(f_e × mean(p_e))`, where `f_e` is the detached share of top-2 assignments for expert e and `mean(p_e)` is its mean softmax probability over non-padding tokens in the microbatch. The nine layer losses are averaged; coefficient 0.001 is added to the task cross-entropy objective, with microbatches weighted during accumulation. Exactly balanced assignments give an unscaled value of 1. This encourages balance, but does not enforce equal loads. There is no router z-loss, expert bias balancing, noise injection or shared-expert auxiliary path.

**What each of the 36 experts actually does: measured evidence**

Every expert is designed to transform a contextual hidden vector for next-token prediction. No expert was assigned 'math', 'code', 'Hindi', 'grammar' or any other named role. All four experts in a layer started as copies of the same general FFN. Their final learned functions are defined by their different weights.

The checkpoint audit confirmed all 36 changed from their copied original FFNs, and all 54 within-layer pairs (6 pairs × 9 layers) now have different weights. Relative L2 changes from the copied FFNs range from 22.45% to 40.86%. This establishes weight divergence, not semantic specialization. Uneven usage, input distribution differences and weight decay can all influence weight changes.

The table below contains every expert's full-validation usage. 'Assignment share' divides by the two assignments per token, so four entries per layer sum to 100%. 'Token selection' gives the fraction of tokens that actually use that expert, so four entries sum to 200%. A balanced expert would have 25% assignment share and 50% token selection. Each layer processed 5,220,706 non-padding input tokens and 10,441,412 assignments. The supervised evaluation loss covers 5,049,456 target tokens.

| Layer | Expert | Parameters | Times selected | Assignment share | Tokens selecting it | Weight change from original, relative L2 |
|---:|---|---:|---:|---:|---:|---:|''']
for e in a['experts']:
    lines.append(f"| {e['layer']} | E{e['expert']} | {e['parameters']:,} | {e['validation_assignments']:,} | {e['assignment_share_pct']:.2f}% | {e['token_selection_pct']:.2f}% | {100*e['relative_l2_change_from_original']:.2f}% |")
lines += [r'''
Layer 1 is strongly concentrated: E1 and E2 together account for 98.49% of assignments. E0 and E3 are selected by only 1.79% and 1.23% of tokens respectively; they are underused, not completely unused. No expert has zero selections across the full holdout. Later layers are generally more balanced, especially layers 5–9. Routing imbalance remains a limitation even though held-out loss improved.

`expert_source_usage.csv` records all 360 layer/expert/source combinations (9 × 4 × 10), with exact counts, assignment shares, token-selection rates and selection frequency relative to that expert's overall rate. `architecture_numbers.json` includes, for every expert, the source with its highest relative selection rate. These are routing preferences, not evidence that the expert is good at that source's task. For example, frequent selection on code alone would not demonstrate a 'coding expert'. Semantic claims would require targeted input/output inspection and controlled expert interventions with task-specific scoring; those per-expert functional tests have not been run.

**Measured evidence that the learned routing matters**

All rows below evaluate the same final trained weights on the full holdout. They change only the inference routing policy; all still execute two experts per token.

| Routing policy | Held-out cross-entropy, nats/token |
|---|---:|''']
for r in s['experiments']:
    label={'learned':'Learned top-2 and normalized learned weights','uniform_selected':'Learned top-2, equal weights of 0.5 each','random':f"Random top-2, equal weights, seed {r['seed']}"}[r['mode']]
    lines.append(f"| {label} | {r['cross_entropy_nats']:.6f} |")
lines += [f'''
The three random runs average 3.465751, versus the normal model's {v['cross_entropy_nats']:.6f}. Equal weighting over the learned choices adds 0.086623 nats; random selection with equal weights adds 0.448624 nats. Both the chosen experts and their mixture weights matter to this checkpoint. The random control jointly changes selection and weighting; comparing random routing with equal weighting over learned choices isolates the selection change under equal weights. The interventions do not prove named expert specialties or how a separately trained random-routing model would perform.

The paired dense control finishes at {d['cross_entropy_nats']:.6f}, compared with MoE {v['cross_entropy_nats']:.6f}. All ten source groups favor this MoE in the recorded comparison. This is one training seed and a validation set already used during development, not a multi-seed or untouched-test claim. The retained generation examples remain repetitive and incoherent, despite the improved next-token loss.
''',r'''
**Active/inactive parameters, memory and parallelism**

Yes, sparse active/inactive expert computation is implemented. No, distributed expert parallelism is not implemented.

| Concept | Our run |
|---|---|
| Physical training GPUs | 1: NVIDIA RTX 3070 Laptop GPU, 8 GiB |
| Expert parallel degree | 1; all 36 experts reside on that same GPU |
| Tensor parallel degree | 1; no matrix sharding |
| Pipeline parallel degree | 1; no layer sharding |
| Data parallel degree | 1; no multi-GPU model replicas |
| Cross-GPU all-to-all dispatch/combine | None |
| ZeRO / FSDP optimizer or parameter sharding | None |
| Expert execution | Token gather, expert matrix operations and weighted scatter/add on the same GPU |
| Shared experts | 0; attention/embedding sharing is a different concept |
| Parameter / gradient / Adam moment precision | FP32 |
| Main attention and expert matmul precision | BF16 autocast |
| Router scoring, softmax and mixture-weight normalization | FP32 |
| Weighted expert-output accumulation | FP32 |

The inactive expert weights remain in GPU memory. Training state is sized by total parameters, not by per-token active parameters. With FP32 parameters, FP32 gradients and two FP32 Adam moments, the main persistent training-state estimate is 16 bytes per parameter. No extra FP32 master weight copy is required because model parameters are already FP32.

| State | Exact bytes | MiB |
|---|---:|---:|''']
for label,key in [('All model weights','parameter_bytes'),('All gradients if allocated','gradient_bytes'),('Both Adam moment arrays','adam_moment_bytes'),('Estimated weights + gradients + moments','approximate_training_state_bytes'),('Measured extension peak PyTorch GPU allocation','measured_peak_allocated_bytes')]:
    n=a['memory'][key];lines.append(f'| {label} | {n:,} | {n/2**20:.2f} |')
lines += [f'''
The measured peak is {a['memory']['measured_peak_allocated_bytes']/2**30:.3f} GiB. It includes activations, temporary tensors and training/evaluation allocations beyond the estimated model state. It is PyTorch allocated memory, not the GPU's total display/driver memory or a promise about peak memory at other batch sizes. Sparse computation does not halve total memory, and 50% expert activation does not imply a 2× end-to-end speedup.
''',r'''
**Training settings and lineage**

| Setting | Actual value |
|---|---|
| Original dense pretraining | 50,000,000 supervised targets |
| MoE continuation, first stage | 10,000,000 reused targets |
| MoE continuation, extended stage | 40,000,000 reused targets |
| MoE-specific training | 50,000,000 targets after conversion |
| Total exposure along model lineage | 100,000,000 targets, including original dense pretraining |
| New unique corpus obtained for continuation | None; one further pass over the existing 50M targets |
| Vocabulary / sequence length | 8,192 / 512 |
| Effective batch | 32 sequences |
| Microbatch | 16 sequences |
| Gradient accumulation | Normally 2 microbatches per full update; final partial batch is smaller |
| Maximum input positions per full update | 32 × 512 = 16,384; not all are supervised targets |
| AdamW betas / epsilon | (0.9, 0.95) / 1e-8 |
| Weight decay | 0.1 on matrix parameters, 0 on one-dimensional normalization parameters |
| Gradient norm clipping | 1.0 |
| Auxiliary balance coefficient | 0.001 |
| First continuation learning rate | 30-update warmup to 1e-4, cosine decay to 1e-5 |
| Extension learning rate | 50-update warmup to 2e-4, cosine decay to 2e-5 |
| Optimizer state at conversion | Fresh AdamW for both dense and MoE comparison arms |
| Optimizer state at extension | Preserved from the matching 10M checkpoint; common new LR schedule |
| Extension optimizer updates | 2,622 per arm |
| Training seed / saved sample order | 20260926 |
| Held-out supervised targets | 5,049,456 |

The model is a small research/assignment demonstration of dense-to-MoE upcycling. Its measured success is continued reduction in held-out next-token loss and dependence on learned routing. Its measured limitations include strong first-layer imbalance and poor generated-text quality.

**Files and reproducibility**

- `architecture_numbers.json`: actual configuration, every named parameter tensor and shape, all per-layer budgets, all 36 expert records, pairwise weight differences, memory calculations and checkpoint identity.
- `expert_source_usage.csv`: all 360 per-source expert usage records.
- `inspect_architecture.py`: read-only checkpoint inventory and consistency assertions.
- `../Session_14_MoE_Extended/moe_model.py` and `dense_model.py`: actual architecture definitions.
- `../Session_14_MoE_Extended/moe/final.pt`: audited final checkpoint.
- `../Session_14_MoE_Extended/moe/final_validation.json`: recorded full-holdout counts and task loss.
- `../Session_14_MoE_Extended/diagnostics/learned.json`: recorded source-conditioned routing.
- `../Session_14_MoE_Extended/diagnostics/summary.json`: routing intervention results.
- `../Session_14_MoE_Extended/GENERATIONS.md`: all retained generation examples.
- Reference document: `C:\Users\udisi\Downloads\session 14.pdf`, pages 3 and 5–8 for reference dimensions and terminology.
''',f"Final checkpoint SHA-256: `{a['checkpoint_sha256']}`. All numeric inventory assertions passed. No trained weights, existing experiments, source data or GPU environment were modified by this architecture audit.\n"]
(ROOT/'ARCHITECTURE.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote ARCHITECTURE.md')
