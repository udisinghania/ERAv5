**Session 14 explained through our actual assignment**

This guide maps the complete 53-page session PDF to the trained assignment. It separates exact architecture calculations, recorded measurements, illustrative estimates, unimplemented features and questions that remain unmeasured. The existing `ARCHITECTURE.md` contains the full architecture table, every expert's usage and all component parameter counts. This guide adds the missing lesson-derived quantities and explains how the pieces fit together.

**What the assignment actually asks**

Page 52 asks for a linear model converted into an MoE, with evidence that training continues and loss decreases; model size and data are left to the student. We interpreted 'Linear model' as the conventional dense Transformer used throughout the lesson, not a standalone linear regression or linear classifier. The submission should state that interpretation.

Our evidence is a pretrained dense Transformer, a full-copy FFN-to-MoE conversion, a conversion-equivalence check, further MoE training, full validation results and a dense continuation control. The MoE's held-out loss went from approximately 3.5142 immediately after conversion to 3.4171 after 10M continuation targets and 3.0171 after 50M. The dense control ends at 3.0779. This satisfies the demonstrated learning objective under that interpretation. The lesson's large-cluster configurations and V5 design discussion are not additional mandatory assignment requirements.

**Numbers missing or needing greater emphasis in the earlier explanation**

| Quantity | Our number | Evidence/type |
|---|---:|---|
| Expert sparsity: stored experts / selected experts | 4 / 2 = **2** | Exact architecture |
| Whole-model total / active ratio | **1.726056** | Exact architecture |
| Parameters inside expert FFNs | **84.1289%** of total | Exact architecture |
| Parameters inside routers | **0.025279%** of total | Exact architecture |
| Expert width / hidden size | 1,664 / 384 = **4.3333** | Exact architecture |
| Active aggregate expert width | 2 × 1,664 = **3,328** per layer | Exact architecture |
| Stored aggregate expert width | 4 × 1,664 = **6,656** per layer | Exact architecture |
| Active width relative to original dense FFN | **2×** | Exact architecture |
| Unordered expert pairs in one layer | C(4,2) = **6** | Combinatorics |
| Theoretical pair paths across nine layers | 6^9 = **10,077,696** | Combinatorial upper bound, not observed paths |
| Routed output scaling factor | **1.0** | No extra multiplier in code |
| Router initialization | Normal(mean **0**, standard deviation **0.01**) | Code |
| Router bias parameters / balance-controller biases | **0 / 0** | Code |
| Router z-loss coefficient | **0**; z-loss not implemented | Code |
| Load-balancing scope | **16 sequences per microbatch**, up to 8,192 input positions | Code; non-padding positions only |
| Effective update batch | **32 sequences**, normally two microbatches | Code |
| Balanced expert load for one full 512-token sequence | **256 tokens/expert/layer** | Assumes no padding |
| Balanced expert load per full microbatch | **4,096 tokens/expert/layer** | Assumes no padding |
| Balanced expert load summed across a full optimizer update | **8,192 tokens/expert/layer** | Across two separate dispatches |
| Capacity with factor 1.25, if imposed per full microbatch | **5,120 slots/expert/layer** | Hypothetical; not imposed |
| Actual overflow-dropped assignments | **0** | Dropless dispatch design |
| Actual full-validation expert assignments, all layers | **93,972,708** | Recorded counts |
| Layer 1 maximum violation | **0.994300** | Recorded holdout load |
| Layer 1 busiest expert / average expert load | **1.994300×** | Recorded holdout load |
| Maximum possible MaxVio for our top-2/4 design | **1.0** | Distinct choices, no dropping |
| Balanced auxiliary loss, before/after coefficient | **1 / 0.001** | Exact loss convention |
| Same-pair collapse auxiliary limit, before/after coefficient | **2 / 0.002** | Selection always same pair, probability mass on it |
| Training work proxy, 6 × active parameters | **0.190094 GFLOP/token** | Approximation, not a profiler result |
| Dense training work proxy | **0.121001 GFLOP/token** | Same approximation |
| 50M continuation work proxy for MoE | **9.504691 × 10^15 FLOPs** | Approximation; not exact masked-data compute |
| Extension throughput, MoE | **35,522 supervised targets/s** | End-to-end recorded time |
| Extension throughput, dense | **57,520 supervised targets/s** | End-to-end recorded time |
| MoE / dense extension elapsed time | **1.6193×** | 1,126.062 s / 695.406 s |
| Actual cross-GPU expert traffic | **0 bytes** | Single GPU |
| BF16 KV cache, if implemented | **13,824 bytes/token = 13.5 KiB** | Hypothetical architecture calculation |
| Same hypothetical cache for 512 tokens | **6.75 MiB per sequence** | Hypothetical |
| Same hypothetical cache for 16 sequences | **108 MiB** | Hypothetical |

The compute proxy follows the PDF's 6P rule. Attention-score operations, router/dispatch overhead, non-padding context positions with masked targets and the selective output head make actual operation counts differ. It must not be labelled measured FLOPs or used to claim hardware utilization. Recorded throughput includes periodic validation and checkpoint saving; the resumed MoE time excludes downtime and discarded uncommitted work.

**A complete map from the lesson to our assignment**

| PDF section | Pages | What it teaches | Our assignment connection |
|---|---|---|---|
| 0. Why this session exists | 1–2 | Store many FFNs; run only a few | Four stored, two executed per token per layer |
| 1. Introduction | 3–6 | Architecture dimensions; total vs active | 9 layers, hidden 384, 54.69M total / 31.68M active; reference Qwen numbers are not ours |
| 2. Terminology | 6–8 | Expert, router, top-k, capacity and parallelism | 36 independent FFNs, nine routers, top-2, no shared experts, EP=1 |
| 3. The Mixture-of-Experts Layer | 8–10 | Attention followed by routed FFNs and weighted combination | Pre-LN attention is preserved; FFN replaced in every block |
| 4. What an Expert Is | 10–11 | One FFN, not a complete model or assigned topic | GELU expert with two matrices, 384→1,664→384 |
| 5. Why Mixture-of-Experts | 12–13 | Capacity/compute tradeoff and costs | More stored capacity; our full-copy top-2 also increases active compute versus dense |
| 6. Active and Total Parameters | 13–16 | Compute, weight/state memory and sparsity | Expert sparsity 2; whole-model total/active 1.726 |
| 7. The Router | 17–19 | Logits, score function, top-k, normalization and precision | FP32 softmax, top-2, renormalization, scale 1, init std 0.01 |
| 8. Expert Size and Shared Experts | 20–21 | Fine-grained experts and always-on shared FFNs | Full-sized copied FFNs; no fine-grained partitioning or shared expert |
| 9. What Experts Learn | 22–23 | Emergent token-pattern or task specialization | Weight divergence and source usage measured; functional specialties not established |
| 10. Load Imbalance and Collapse | 24–25 | Feedback loop, dead experts and MaxVio | Zero dead experts over full holdout, but layer 1 is near maximal load imbalance |
| 11. Capacity and Token Dropping | 25–27 | Slot limits versus dropless computation | Dropless Python dispatch, not a grouped/block-sparse kernel |
| 12. Auxiliary Losses | 28–29 | Balancing loss and router z-loss | Balance coefficient 0.001; no z-loss |
| 13. Auxiliary-Loss-Free Balancing | 29–32 | Selection-only expert biases and load controllers | Not implemented; no bias update rate or controller |
| 14. Balancing Scope | 33–35 | Sequence, microbatch and whole-batch load accounting | Microbatch balancing over 16 sequences, not whole effective batch |
| 15. Growing a Mixture-of-Experts Model | 36–39 | Copy, partition, drop-upcycling and clone collapse | Full-copy upcycling; no redrawn neurons, no later expert growth |
| 16. Expert Parallelism | 40–43 | Distribute experts, dispatch/combine token vectors | Concept only here; all experts are on one GPU |
| 17. Expert Parallelism with the Other Forms | 43–45 | Dense/expert replication, ZeRO, activations and overlap | No distributed sharding, no communication overlap or expert replicas |
| 18. Case Studies (self study) | 45–47 | Compare architecture and systems design choices | Reference examples, not components automatically present in our model |
| 19. Choosing a Mixture-of-Experts Layout | 47–50 | Count state, activations, token load, traffic and throughput | Single-GPU measured peak 3.987 GiB; timings measured; detailed profiler breakdown absent |
| 20. V5 Decisions | 51–52 | Possible course-model design choices | Bias balancing, EP=8 and RL router freezing are not used or required here |
| 21. Assignment | 52 | Convert dense model and demonstrate further learning | Conversion and held-out loss reduction recorded |
| Studio/navigation | 53 | Embedded video/navigation | No further numeric assignment requirement in the PDF |

The PDF contains static exports of several interactive lesson figures; some interactive areas are blank. Our visual walkthrough reconstructs the relevant mechanisms and uses our saved measurements where labelled. Claims about other models in the PDF remain the PDF's examples, not results from this experiment.

**Follow one token through the model**

Start with a token ID. Its learned embedding has 384 numbers, and its learned position vector also has 384 numbers. We add them. This vector passes through nine blocks. In every block, LayerNorm and causal attention let it incorporate earlier tokens from its own document. Another LayerNorm prepares the resulting contextual vector for the router.

The router's 384×4 matrix produces four logits. Softmax converts them into probabilities that sum to one. Top-2 chooses two distinct experts. The selected probabilities are divided by their sum, and these two mixture weights also sum to one. The two experts each transform a 384-number vector into 1,664 hidden activations and back to 384. We multiply their outputs by their mixture weights, add them and then add the block's residual input. The next layer repeats with a different router and different expert bank.

After the final block, a final LayerNorm and the tied 384→8,192 vocabulary projection produce next-token logits. An expert output is therefore not itself a word prediction. The final vocabulary head makes that prediction using the representation produced by all nine blocks.

`token_routing_trace.json` captures actual logits, probabilities, selected experts and mixture weights from the final checkpoint for two prompts: an arithmetic sentence and a Python function. The interactive routing view lets you change token position and layer. Tokenization uses pieces, so a visible word can occupy several token positions. A special beginning token is also present. These short prompt traces illustrate actual execution; they do not establish expert specialties or broad routing statistics.

**Tensor sizes at a full microbatch**

For B=16, T=512 and H=384, assuming no padding:

| Tensor or operation | Shape / count | Interpretation |
|---|---|---|
| Token/hidden representation | [16, 512, 384] | One 384-vector per input position |
| Combined QKV projection | [16, 512, 1,152] | Q, K and V together |
| Each of Q, K and V after reshaping | [16, 6, 512, 64] | Six heads, each 64 wide |
| Router logits per layer | [8,192, 4] | 32,768 FP32 values, 128 KiB |
| Selected expert IDs per layer | [8,192, 2] | 16,384 int64 IDs, 128 KiB |
| Selected mixture weights per layer | [8,192, 2] | 16,384 FP32 values, 64 KiB |
| One expert's input | [n_e, 384] | n_e depends on routing, not a fixed capacity |
| One expert's intermediate activation | [n_e, 1,664] | GELU applies here |
| Sum of expert input rows in a layer | 16,384 | Two assignments for each of 8,192 tokens |
| Sum of expert intermediate elements | 27,262,976 | 52 MiB if represented once in BF16; not a peak-memory estimate |
| Combined MoE output | [16, 512, 384] | Weighted accumulation is FP32 |

The actual implementation selects non-padding rows, so padded batches can be smaller. The attention backend need not materialize a full attention score matrix. Training computes vocabulary logits only at supervised prediction positions. These details are why adding a few tensor sizes or copying the PDF's activation-memory coefficient does not reproduce measured peak allocation.

**Why total capacity and per-token activity differ**

All 36 expert FFNs exist in memory. One token selects two in each layer, or 18 expert applications across the model. The other 18 are skipped for that token, but another token can select them. This is conditional computation. It is distinct from freezing parameters and distinct from placing experts on different GPUs.

Our count is:

`total = 8,679,168 common parameters + 36 × 1,277,952 expert parameters = 54,685,440`

`active = 8,679,168 common parameters + 18 × 1,277,952 expert parameters = 31,682,304`

Full embedding tables are included in this architectural convention. Lookup positions do not literally read every row, while the tied output vocabulary projection uses the token embedding matrix at predicted positions. Parameter counts should not be confused with memory bandwidth or exact per-position FLOPs.

Changing E while keeping k fixed mainly changes stored capacity. Changing k while keeping E fixed changes how much expert computation each token receives. Changing E also changes the router's parameter count slightly. The interactive accounting view includes that small router adjustment.

| Top-k, with four stored experts | Active parameters | Aggregate active FFN width | Trained in this assignment? |
|---:|---:|---:|---|
| 1 | 20,180,736 | 1,664 | No |
| 2 | 31,682,304 | 3,328 | Yes |
| 3 | 43,183,872 | 4,992 | No |
| 4 | 54,685,440 | 6,656 | No |

These are counterfactual architecture counts, not loss predictions. Changing top-k at inference can damage a model trained for top-2. There is also a learning subtlety: with exactly one selected expert and normalization over that one weight, its weight becomes exactly 1. The current implementation would then lose the usual task-gradient path through mixture weights to the router, although the auxiliary balancing gradient remains. A top-1 experiment needs careful training design; it is not automatically a free speedup.

The PDF's fine-grained example uses several small experts whose combined width matches a dense FFN. Our copy-based design instead executes two full-width FFNs. It demonstrates upcycling and sparse routing, but does not demonstrate matching or beating dense compute at the same active width.

**How selection differs from weighting**

The selected IDs determine which functions run. Their weights determine how much the selected outputs contribute. These are separate decisions. Without selection biases, softmax, sigmoid and softplus preserve the ordering of the same logits, so they choose the same top-k IDs; their normalized output weights can still differ. Our implementation uses softmax only, followed by selected-score normalization and no additional scale (effective scale 1).

The router runs in FP32 to keep scoring and normalization numerically stable. This does not mean the entire model trains in FP32 arithmetic: attention and expert matrix multiplications use BF16 autocast, while weights, optimizer state and router arithmetic remain FP32. FP32 routing and z-loss address related stability concerns but are not the same mechanism. We use the former and do not implement the latter.

**How the experts learn, and what 'expert' does not tell us**

At conversion, each dense FFN is copied four times with no partition, noise injection or neuron redrawing. Attention, embeddings and norms are copied too; only router weights are new. Because the mixture weights sum to one, mixing identical expert outputs initially reproduces the dense FFN. Once different tokens are routed to different copies, gradients make their weights differ.

The task loss updates selected experts and the shared parts of the Transformer. It updates router weights through the selected continuous mixture weights; it does not differentiate through the discrete ID choice. The balancing loss provides an additional router gradient. For a particular token, unselected experts receive no task gradient from that token. They can still learn from other tokens, and optimizer momentum/weight decay can move parameters with zero current task gradient when those parameters participate in an optimizer step.

All 36 experts changed from the copied source FFN. The largest and smallest relative L2 changes are 40.86% and 22.45%; every within-layer pair now differs. That proves parameter divergence, not a named capability. The PDF discusses token classes, syntax, languages, subjects and especially influential experts as empirical findings from different studies. None of those semantic labels can be assigned to our E0–E3 merely from their numeric IDs, source usage or weight norms.

Our experiments do show dependence on routing: learned routing gives loss 3.017127; equal weights over the learned choices give 3.103750; random top-2 with equal weights gives mean 3.465751 over three inference seeds. Random routing changes selection and weighting together; comparison with the equal-weight learned-choice control isolates the selection change under equal weights. These are inference interventions without retraining, not quality estimates for independently trained alternative designs.

**Why load collapse happens and how to read MaxVio**

A slightly preferred expert receives more routed tokens, learns on those tokens and may become even more preferred. Other experts get fewer opportunities. Auxiliary balancing attempts to counter this feedback. Some concentration can coexist with better task loss, and balancing pressure can compete with the task objective.

For assignment counts c_e, `MaxVio = max(c_e) / mean(c_e) - 1`. A value of 0 means equal counts; a value of 1 means the busiest expert has twice the average.

| Layer | MaxVio | Busiest / average | Full-holdout unused experts |
|---:|---:|---:|---:|
| 1 | 0.994300 | 1.994300× | 0 |
| 2 | 0.442215 | 1.442215× | 0 |
| 3 | 0.629483 | 1.629483× | 0 |
| 4 | 0.478853 | 1.478853× | 0 |
| 5 | 0.125937 | 1.125937× | 0 |
| 6 | 0.142100 | 1.142100× | 0 |
| 7 | 0.091217 | 1.091217× | 0 |
| 8 | 0.123737 | 1.123737× | 0 |
| 9 | 0.127072 | 1.127072× | 0 |

Our top-2 choices are distinct. In a layer with N input tokens, one expert can receive at most N selections, while the average is 2N/4=N/2. Therefore MaxVio cannot exceed 1 for this architecture. Layer 1 at 0.9943 is near this limit. It must not be compared numerically with a 128-expert, top-1 example without accounting for that design's much higher maximum.

Layer 1's two busiest experts grow from 79.83% of assignments after 10M continuation targets to 98.49% after 50M. MaxVio alone barely changes because E1 was already very busy; concentration of the second expert reveals the additional collapse. As a complementary descriptive statistic, the effective number of experts from assignment entropy is 2.162 in final layer 1, versus approximately 3.98 in layers 5, 7, 8 and 9. This describes aggregate load diversity, not the two experts active on each token and not per-token router uncertainty.

'Zero dead experts' means every expert received at least one selection over the full holdout. It does not say every expert was selected in every training microbatch or that all were well trained. We have not computed a per-microbatch dead-expert rate.

**Capacity, token dropping and microbatch scope**

With N non-padding tokens, E experts and k selections, balanced load is Nk/E. For our full microbatch, this is 8,192×2/4=4,096 token assignments per expert. A hypothetical capacity factor of 1.25 gives 5,120 slots. A much busier expert could exceed those slots and lose assignments in a capped implementation.

Our code sets no such cap. It gathers every selected token, executes each expert on however many rows arrive, and combines outputs. Overflow dropping is zero by construction. Padding exclusion and supervised loss masking are separate concepts; they are not overflow dropping. The code loops over experts, so dropless execution does not imply use of MegaBlocks or an optimized grouped GEMM kernel.

The balancing loss is evaluated on each microbatch of 16 sequences separately, averaged across nine layers and weighted during gradient accumulation. Combining two microbatch gradients to obtain an effective batch of 32 does not turn the loss into a balancing objective computed from whole-batch probability/count aggregates. That distinction matters because different microbatches may contain different token or source distributions. No experiment has established whether whole-batch balancing would improve our model.

Our exact per-layer convention is `L_balance = 4 × sum(f_e × mean(p_e))`, with f_e the share of all top-2 assignments and with its gradient detached. The training coefficient is 0.001. Uniform assignment shares make unscaled loss 1. If every token chooses the same two experts and probability mass concentrates on that pair, the loss approaches 2, not 4. The PDF's top-1 example in which all tokens choose one expert cannot be copied literally into our distinct top-2 design.

**Balancing methods in the PDF that we do not use**

- A selection-only bias controller adds a per-expert bias when choosing IDs but obtains mixture weights from the original scores. Busy experts receive a reduced selection bias, idle ones an increased bias. Our model has no such biases or update speed.
- Router z-loss penalizes the squared log-sum-exp of logits. Its coefficient is zero here because it is not implemented.
- Sequence-level auxiliary balancing is not separately added. Our one balancing term operates over non-padding microbatch tokens.
- Probabilistic or noisy early routing can give copied experts more opportunities. Our training uses hard top-2 throughout.
- Router freezing and routing replay during reinforcement learning are discussed in the case studies. Our experiment uses supervised next-token training, with no RL phase and no frozen router.

These omissions are explicit design choices/limitations, not unreported required assignment numbers. Their effects would need separate experiments.

**Expert parallelism: where it would fit**

Sparse selection answers 'which experts run?'; expert parallelism answers 'which GPU owns them?'. Our answers are 'two of four per layer' and 'all on the same GPU'. EP, TP, PP and DP degrees are all 1. There is no inter-GPU expert dispatch/combine traffic, expert replication, gradient all-reduce, ZeRO sharding, DualPipe overlap or dynamic expert load-balancing placement.

A hypothetical two-GPU layout could put E0/E1 on GPU 0 and E2/E3 on GPU 1 in every layer. A token whose attention runs on GPU 0 and chooses E1/E3 would execute E1 locally, send its vector to GPU 1 for E3, receive E3's output, and combine both outputs at GPU 0. Backpropagation sends corresponding gradient information. The real assignment does none of these network transfers; its gather/scatter stays on one GPU.

The PDF's rough communication formula is `4 × tokens × layers × k × hidden × bytes_per_value × remote_fraction`, counting dispatch/combine and forward/backward. For our EP=1, the remote fraction is zero. A hypothetical balanced EP=2, BF16 payload, one 512-token sequence gives 13.5 MiB per source GPU using that simplified formula. This is not measured traffic, omits metadata/implementation details, assumes half the selections remote and uses 16-bit payloads; our current implementation has FP32 accumulation and no communication serialization.

With EP=2, an even split would give two experts per layer, or 18 expert modules, to each GPU. Attention and other common parameters could be replicated, and optimizer state might be sharded separately if implemented. This reduces per-GPU expert memory but introduces communication and load imbalance. No speedup can be asserted from the split alone.

**Memory, KV cache, activations and compute are different budgets**

All FP32 model weights use 218,741,760 bytes (208.61 MiB). Adding FP32 gradients and two Adam moment arrays produces a main training-state estimate of 874,967,040 bytes (834.43 MiB), excluding small optimizer metadata and temporary allocations. We do not require an extra FP32 master copy because weights already are FP32. The measured extension peak PyTorch allocation is 4,281,106,944 bytes (3.987 GiB).

The gap includes activations, attention/backend temporaries, logits, gradient computation and optimizer/evaluation allocations. It has not been profiled into components. It would be inaccurate to call the whole gap 'activation memory' or to transfer the PDF's 34-bytes-per-hidden-unit estimate directly to this architecture.

A KV cache is a separate inference optimization. It retains earlier attention keys and values to avoid recomputing them at each generated token. Our helper does not implement it. If a BF16 cache were added to our six-KV-head, 64-head-dimension, nine-layer model, it would require `2 × 6 × 64 × 2 × 9 = 13,824 bytes/token`, or 6.75 MiB for 512 tokens per sequence. This is hypothetical cache payload, not current memory use; it excludes the rest of inference memory. A BF16 cache for 16 full sequences would be 108 MiB.

The PDF's 6P rule estimates dominant training arithmetic. It gives 0.190094 GFLOP/token for the MoE versus 0.121001 for the original-shape dense model. Actual measured extension timing is 18.77 minutes for MoE and 11.59 minutes for dense, with equal 40M supervised targets. Those times include evaluations and saves. The MoE is 1.6193× slower in this recorded stage; higher capacity and lower loss did not make this implementation faster than dense.

**What is established, and what remains unknown**

Established: checkpoint architecture/counts, all experts' weight divergence, full-holdout expert usage, layer imbalance, exact training target accounting, loss reduction, dependence on routing, end-to-end timing and peak allocated GPU memory. The actual prompt traces add per-token routing examples without changing weights.

Not established: semantic roles for all experts; task-specific expert ablation scores; full-dataset router entropy or top-k margin distributions; per-microbatch starvation; retained activation-memory breakdown; measured FLOPs/MFU; latency with a KV cache; multi-GPU performance; quality of top-1, shared-expert, finer-grained, bias-balanced or whole-batch-balanced alternatives; variability across independent training seeds. These are missing measurements, not numbers that can be inferred from parameter counts.

The current model still produces repetitive/incoherent samples. A lower next-token loss is meaningful evidence of learning, but not evidence that a useful general assistant or named subject specialists emerged.

**Check your understanding**

1. Why does a 54.69M model use only 31.68M active parameters? Each token uses two of four experts per layer; common parameters are included in both counts.
2. Does an inactive expert disappear? No. It stays in memory and can be selected by other tokens.
3. Is E0 one expert reused nine times? No. Every layer has a separate E0 with separate weights.
4. Does top-2 mean two experts for the entire model? No. Two in each layer, giving 18 expert applications across nine layers.
5. Does zero dead experts mean balance? No. Layer 1 uses all four somewhere but allocates 98.49% of assignments to two.
6. Does gradient accumulation make balancing global? No. Our balancing loss is computed separately on each microbatch.
7. Does having four experts mean four GPUs? No. Sparse selection and expert placement are independent choices.
8. Why are there no guaranteed 'math' or 'code' experts? Roles are not assigned, and routing frequency is not a functional capability test.
9. Can we halve memory by switching top-2 to top-1? No. All four experts and their training state are still stored; the architecture changes active work.
10. What does the assignment prove? Our dense-to-MoE conversion continues learning and lowers held-out loss; it does not prove compute-matched superiority or complete language competence.

**Evidence files**

- `ARCHITECTURE.md`: full model specification and exact component budget.
- `architecture_numbers.json`: checkpoint-audited tensors, parameters and expert changes.
- `session_audit_numbers.json`: all added lesson-derived calculations and qualifications.
- `token_routing_trace.json`: actual prompt tokens and all nine layers' router outputs.
- `expert_source_usage.csv`: all 360 expert/source records.
- `visual_checks.json`: layout and interaction checks for the visual explanations.
- `../evidence/stage2/REPORT.md`: training and control experiment results.

Source: the supplied `Session 14 course PDF (provided separately)`, all 53 pages. No existing checkpoint, experiment, dataset or training environment was changed for this review.
