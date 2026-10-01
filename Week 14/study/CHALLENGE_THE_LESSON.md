**Challenge the MoE lesson: a technical cross-examination of the PDF and our assignment**

Use this alongside `SESSION_STUDY_GUIDE.md` and `ARCHITECTURE.md`. The goal is to identify what is a definition, what follows mathematically, what is an empirical observation, and what is an engineering choice. A useful objection proposes a competing explanation and a way to distinguish it. It does not replace evidence with automatic disagreement.

Each discussion gives the idea and its mechanism, the strongest objection, a qualified answer, and a test or consequence for our assignment. Page numbers refer to the supplied 53-page session PDF. External links are primary research checked for this critique. The recent-model case-study catalogue has not been independently audited model by model; none of its headline numbers is treated here as proof of a universal rule.

**1. Why use MoE instead of one dense model? (Sections 0, 5; pp. 1–2, 12–13)**

The idea is conditional computation: store multiple transformations but apply only a subset to each input. This can separate stored capacity from computation per token.

**Skeptic:** Why not spend the same memory and engineering effort on a dense model, more data or longer training?

**Answer:** That is a valid competing design. MoE is attractive only if its quality/cost tradeoff is better under the resource constraint that matters. Equal tokens, equal FLOPs, equal memory and equal elapsed time answer different questions. A small active model with large stored capacity can still be expensive to serve or difficult to optimize.

**Our evidence:** MoE loss 3.0171 beats dense 3.0779 at equal continuation targets. But MoE has 31.68M active parameters versus dense 20.17M and takes 1.619× the extension time. We have not demonstrated better quality at equal compute, time or memory. A compute/time-matched dense continuation and an active-width-matched dense FFN are stronger controls.

**2. Why replace the FFN rather than attention? (Sections 1, 3; pp. 3–10)**

The FFN already maps each contextual token vector independently back to the same hidden dimension. Replacing it with several shape-compatible FFNs fits the residual block and gives a convenient place to add conditional capacity. Attention still supplies information from other tokens.

**Skeptic:** Is attention incapable of being conditional, or does the FFN contain all the knowledge?

**Answer:** Neither follows. FFN replacement is a practical architecture choice. Parameters and computation are also present in attention and embeddings, and useful behavior is distributed across their composition. Routing attention would require a different design and different cache/communication considerations; the lesson's FFN-MoE example does not establish that other designs are impossible.

**Our evidence:** Attention is preserved in each block. Expert weights account for 84.13% of total parameters. That percentage is an inventory fact, not a measurement that experts contain 84.13% of knowledge.

**3. Why these layer, head and hidden sizes? Why not copy Qwen? (Section 1; pp. 3–5)**

Hidden size controls the token representation; heads divide attention projections into separate subspaces; depth composes successive transformations. Our six heads of 64 dimensions fit hidden size 384. The PDF's Qwen example projects to attention dimensions that need not equal its hidden size.

**Skeptic:** Are nine layers, six heads and width 384 somehow optimal?

**Answer:** No optimum was measured. These were inherited from the available trained dense checkpoint and fit our local environment. Copying only a large model's ratios does not copy its training behavior, data requirements or kernel performance. Different numbers can be valid if dimensions and residual interfaces remain compatible.

**Test:** Compare alternative shapes at a declared budget. Calling an inherited shape 'optimal' without such a comparison is unjustified.

**4. Why GQA? Does an eight-times smaller KV cache give eight-times more serving capacity? (Section 1; pp. 3–4)**

GQA shares key/value heads across groups of query heads, reducing stored keys and values. It trades attention flexibility for a smaller cache and different bandwidth requirements. The original [GQA paper](https://arxiv.org/abs/2305.13245) investigates this tradeoff and converting multi-head checkpoints.

**Skeptic:** The PDF's cache reduction sounds like a whole-system speedup. Is it?

**Answer:** A cache reduction factor applies to the cache component. Total memory also includes model weights, runtime buffers and other state; throughput also depends on computation and bandwidth. Therefore an eight-times smaller KV payload does not guarantee eight-times more end-to-end throughput or sequences.

**Our evidence:** Six query heads, six KV heads, ordinary multi-head attention and no implemented KV cache. The 6.75 MiB cache figure for a 512-token sequence is a hypothetical BF16 payload, not measured current memory. GQA would change the architecture and need evaluation.

**5. Why call these networks experts? (Sections 2, 4; pp. 6–11)**

An expert is an independently parameterized FFN selected by a routing rule. The name describes its role in conditional computation, not a certified subject specialty.

**Skeptic:** Four small FFNs are not four intelligent agents. Why imply expertise?

**Answer:** Correct: these are mathematical functions, not people, autonomous agents or complete language models. Each of ours computes `W_down GELU(W_up x)`. The input x already contains context supplied by attention. Named specialties require separate evidence.

**Our evidence:** E0 in one layer is unrelated by identity to E0 in another layer. All 36 use the same architecture but separate weights. No expert was assigned a subject or source label.

**6. Why GELU, two matrices and width 1,664? Why not SwiGLU? (Sections 3–4; pp. 8–11)**

Our expert preserves the source FFN: a 384→1,664 matrix, GELU and a 1,664→384 matrix. Reusing that function is what makes straightforward copy-based conversion possible.

**Skeptic:** The PDF uses a three-matrix gated FFN. Did we omit a necessary part of MoE?

**Answer:** No. MoE requires a mixture of expert functions; a particular expert activation is not part of its definition. Replacing GELU with a gated FFN would change the function, parameter budget and initialization problem. It could be beneficial, but it would not be a neutral implementation cleanup.

**Test:** Compare activations at matched active parameter count and training budget. Keep our exact conversion experiment separate from such an architecture experiment.

**7. Why sum expert outputs? Is this just an ensemble? (Section 3; pp. 8–10)**

Each expert returns the same hidden dimension, so a weighted sum can be added through the residual connection. Routing makes this input-dependent and sparse at every layer.

**Skeptic:** Why not concatenate the outputs, average them or use one large FFN?

**Answer:** Those define alternative models. Concatenation requires an additional projection or a wider downstream representation. Equal averaging removes learned weighting. A conventional ensemble often combines predictions from separate complete models; our experts share the surrounding Transformer and combine intermediate features.

**Our evidence:** Equal weights over the learned expert choices worsen loss from 3.0171 to 3.1037. That establishes that the trained weighting matters to this checkpoint. It does not prove a separately trained equal-weight design would lose.

**8. Why top-2? Why not top-1, all experts or adaptive k? (Sections 3, 7; pp. 8–10, 17–19)**

Top-k fixes the number of expert applications per token, making the expert compute budget predictable. Our k=2 allows two complementary outputs to contribute while skipping half the bank.

**Skeptic:** Was two selected because it was best?

**Answer:** No k sweep was run. Four experts/top-2 was a small, feasible experiment. All four increases active expert work; top-1 reduces it but changes the learning path. Adaptive k could allocate differing amounts of computation but adds another decision and variable load.

**Critical detail:** Our selected probabilities are normalized to sum to one. At k=1, the only selected weight is p/p=1, so its task derivative through that weight is zero away from routing boundaries. This is a property of our normalization, not a theorem that top-1 routing cannot learn. [Switch Transformer](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf), Section 2.1, retains the selected softmax gate value rather than renormalizing it to one. The numeric gradient examples in `critical_examples.json` verify the distinction.

**9. Why should tokens choose experts? Could experts choose tokens? (Section 7 and routing design)**

Token-choice routing gives each token a fixed number of expert assignments. Reversing the assignment can regulate expert workload directly.

**Skeptic:** We created a balancing problem by letting every token choose the same favorites. Why not reverse the choice?

**Answer:** That is an actual alternative. [Expert Choice Routing](https://arxiv.org/abs/2202.09368) lets experts select token buckets, giving tokens potentially different numbers of experts. The tradeoff is controlling expert load versus controlling each token's coverage and compute. In an autoregressive implementation, a selection that depends on other sequence positions also needs a careful causality check; it must not leak future information.

**Our evidence:** We implement per-token top-2 with causal attention. No expert-choice comparison was run. Constant k is our design, not the only definition of MoE.

**10. Why softmax? Is its output expert confidence? (Section 7; pp. 17–19)**

Softmax converts logits to positive scores. We select two and renormalize them, obtaining a convex combination of expert outputs.

**Skeptic:** Does a 90% routing probability mean the expert is correct 90% of the time? Why not sigmoid?

**Answer:** Routing weights are not calibrated probabilities of correctness. They are learned control values used to optimize the final task. With the same logits and no selection biases, softmax, sigmoid and softplus preserve rank, so top-k IDs agree; normalized weights can differ. Across separately trained systems their learned logits and selected IDs can of course differ too.

**Our consequence:** For a fixed selected set S, normalized softmax weights simplify to `exp(z_i)/sum_{j in S} exp(z_j)`. Unselected logits cancel from this local task-weighting expression; auxiliary balancing still depends on probabilities over all experts. This explains both useful continuous gradients and the discrete blind spot at selection boundaries.

**11. Why normalize weights to one? Why a scaling factor? (Sections 3, 7; pp. 9, 18)**

Normalization controls mixture scale and makes mixing identical copies reproduce their shared output. A routed scaling factor can alter the MoE branch's magnitude relative to the residual stream.

**Skeptic:** Does normalization always give the right output magnitude? Is scale 1 more correct than 2.5?

**Answer:** Neither is a mathematical optimum for task quality. Expert output norms, covariance and k affect the mixture. For independent zero-mean scalar outputs of variance v, equal averaging over k gives variance v/k; identical outputs instead retain variance v. A scale factor compensates differently in those two situations. Real expert outputs need not satisfy either simplifying assumption.

**Our evidence:** The effective factor is 1. Changing it would alter a trained function and needs validation. A larger model's scale constant cannot be copied as a universal recipe.

**12. Why FP32 routing and small initialization? (Section 7; pp. 18–19)**

Router decisions depend on numerical score ordering; scores close to a top-k boundary can change membership under rounding. FP32 improves precision at relatively low parameter cost.

**Skeptic:** Does FP32 guarantee stable training? Should all logits remain near zero forever?

**Answer:** No. Finite but bad optimization, large activations or unsuitable learning rates can still fail. Small initialization controls the starting distribution, not the desired final uncertainty. [ST-MoE](https://arxiv.org/html/2202.08906v2), Section 3.3, reports that selective FP32 alone was insufficient at its largest scales and investigates an additional router z-loss.

**Our evidence:** Router initialization std is 0.01, router computation is FP32 and the run remained finite. No initialization or precision ablation proves these values optimal for our model.

**13. Why count active parameters? Does that determine runtime? (Sections 5–6; pp. 12–16)**

Active parameter count approximates how much weight-dependent arithmetic a token uses, especially the large matrix operations.

**Skeptic:** Same active count means same speed, right?

**Answer:** No. Routing, memory access, kernel launch overhead, matrix shapes, attention length, communication and batch composition also affect runtime. Even the standard count includes full embedding tables despite row lookups, while the vocabulary projection has its own position-dependent use. The PDF's 6P training estimate is a useful arithmetic proxy, not a profiler measurement.

**Our evidence:** The MoE has 1.571× the dense active parameter count and took 1.619× its extension elapsed time. Their numerical proximity here does not establish a universal runtime law; the measurements include evaluation and checkpoint saving.

**14. Why keep inactive weights in memory? Does inactivity mean no learning? (Section 6; pp. 13–16)**

Any expert can be selected by a later token, so our straightforward implementation keeps all experts resident. Gradients and optimizer moments are also sized by the stored parameters during training.

**Skeptic:** Could we move idle experts away, or do inactive experts stay permanently frozen?

**Answer:** Offloading, quantization or pruning would change residency, representation or the model itself, with transfer, accuracy or implementation tradeoffs. 'Inactive' only means not selected for this token. Other tokens can train the same expert; momentum and weight decay can also affect parameters with zero current task gradient when the optimizer processes them.

**Our evidence:** All experts are resident on one GPU. Main FP32 weight/gradient/Adam-state accounting is 834.43 MiB, but peak allocation is 3.987 GiB. The difference is not automatically all activation memory: temporary operations, logits, optimizer work and evaluation also contribute.

**15. Why many small experts? Do more combinations imply more intelligence? (Section 8; pp. 20–21)**

Dividing a fixed active width into smaller independently selectable pieces increases the number of possible subsets and can permit finer conditional computation. It also changes router size and kernel shapes.

**Skeptic:** Counting combinations is cheap. What if every expert computes the same thing?

**Answer:** Then many routes can produce the same function. Combinatorial possibility is not learned diversity, actual path usage or quality. Very small experts can also receive too little data per update or make inefficient matrix operations. The optimum is an empirical quality/hardware tradeoff.

**Our evidence:** C(4,2)=6 pairs per layer and 6^9 possible layerwise pair paths are only upper bounds. They are not 10 million independently trained models. A fine-grained candidate for our active-width budget would be eight width-832 experts selecting four, but no result for that candidate exists.

**16. Why shared experts? Does forcing one always on create common knowledge? (Section 8; pp. 20–21)**

An always-on expert offers a path for transformations useful to all inputs. The hypothesis is that routed experts then need less duplicated common functionality.

**Skeptic:** Couldn't attention, residual pathways or frequently chosen routed experts already provide that? Isn't an always-on expert spending compute on every token whether needed or not?

**Answer:** Both are valid tradeoffs. [DeepSeekMoE](https://arxiv.org/abs/2401.06066) proposes shared-expert isolation as part of its design. [OLMoE's shared-expert ablation](https://arxiv.org/html/2409.02060v2#S4.SS1.SSS3) finds no benefit in its tested setup. These are conditional empirical results, not a contradiction that must be resolved by declaring one universally correct.

**Our evidence:** Zero shared experts. A fair first comparison would replace one of two active routed paths with one shared path while matching the active budget. Simply adding a shared expert would confound the architecture choice with extra compute.

**17. What have experts actually learned? Does usage reveal skill? (Section 9; pp. 22–23)**

Routing statistics describe which inputs encounter an expert. Functional tests ask what the expert contributes to the model's behavior.

**Skeptic:** Frequent selection on code might just reflect spaces, punctuation or common tokens. And different weights might compute the same function.

**Answer:** Correct. Source identity is a coarse label and can correlate with token frequency, formatting or sequence position. There are also parameter symmetries: permuting the hidden neurons of a GELU FFN and applying the inverse arrangement to its output matrix preserves the function while changing both matrices. Our numeric example changes weights but reproduces outputs to about 1.8e-15.

**Our evidence:** All 36 experts' weights changed, and routing interventions worsen loss. These establish parameter divergence and dependence on the trained routing arrangement, not named specialties. Better evidence would include controlled token-category comparisons, expert ablations with scale-preserving controls, rare-domain tests and measurements of actual expert outputs.

**18. Do influential experts prove others are disposable? (Section 9; pp. 22–23)**

An expert ablation measures the effect of a specific intervention in the model's current state.

**Skeptic:** A rare expert might serve rare but valuable inputs. Removing any output changes branch scale. How can we attribute the loss to specialized knowledge?

**Answer:** We cannot infer that from frequency or a single uncorrected ablation. Compare removal, replacement by a matched-scale output, rerouting and, separately, retraining after removal. Report performance across rare subgroups, not just a corpus average. Jointly removing two experts can differ from adding their individual effects.

**Our consequence:** Layer 1 E0/E3 are underused but have not been shown redundant. No pruning recommendation follows solely from their low assignment shares.

**19. Why balance expert load at all? Could imbalance be intelligent allocation? (Section 10; pp. 24–25)**

Balancing protects learning opportunities and, under expert parallelism, prevents a heavily loaded device from becoming the bottleneck. But demand for useful functions need not be uniform.

**Skeptic:** If two experts work best, forcing tokens onto others can worsen predictions. Isn't balance an arbitrary preference?

**Answer:** Perfect uniformity is not the task objective. Some skew could be useful specialization, redundancy avoidance or a common-function pathway. Severe skew could instead be self-reinforcing starvation. The observed load alone does not distinguish those explanations. The practical target is a quality/throughput tradeoff with adequate expert training, not the prettiest histogram.

**Our evidence:** Task loss improved while layer 1's top-two assignment share increased to 98.49%. Thus concentration did not prevent aggregate learning in this run. Whether stronger balancing improves quality, rare domains or throughput remains untested.

**20. Why MaxVio and dead-expert counts? What do they miss? (Section 10; pp. 24–25)**

MaxVio is the busiest expert's load divided by average load minus one. Dead-expert counts record zero selections in a specified observation window.

**Skeptic:** A single extreme value cannot describe the whole distribution. A once-used expert isn't healthy either.

**Answer:** Correct. Distributions [50%,25%,25%,0%] and [50%,50%,0%,0%] both have MaxVio 1 but very different breadth of use. Full-holdout zero-use counts can hide starvation within individual training microbatches. GPU ownership also matters: two equally imbalanced expert histograms can load devices differently.

**Our evidence:** MaxVio in layer 1 barely changes from about 0.985 to 0.994 while top-two concentration grows from 79.83% to 98.49%. We need complementary measures. Also, 98.49% of assignments is not necessarily 98.49% of tokens choosing that exact pair; those use different denominators. Aggregate marginal counts do not fully specify pair co-selection.

**21. Why capacity limits? Is dropless always faster and better? (Section 11; pp. 25–27)**

Caps bound per-expert work and simplify fixed-shape execution; excess assignments skip the expert branch. Dropless dispatch retains all selected assignments, with variable expert input sizes.

**Skeptic:** A cap could protect memory and latency. Why dismiss it? And how does retaining every token make irregular work fast?

**Answer:** Droplessness is a correctness/coverage property, not an efficiency guarantee. It avoids one source of lost computation but still needs effective kernels and sufficient memory. [MegaBlocks](https://arxiv.org/abs/2211.15841) specifically develops block-sparse GPU operations to address that implementation problem. Caps may be useful under hard runtime constraints, but dropping changes the model's executed function and must be evaluated.

**Our evidence:** No overflow dropping, but experts are visited in a Python loop, not a MegaBlocks-style grouped kernel. A hypothetical factor 1.25 allows 5,120 assignments per expert in a full microbatch; it is not an actual cap in our run. Padding exclusion and loss masking are not overflow dropping.

**22. Why add a balancing loss? Is a lower balancing loss proof of better balance? (Section 12; pp. 28–29)**

Our surrogate is `4 × sum(f_e P_e)`, where f is the detached assignment share and P is mean router probability. It supplies differentiable pressure via P toward changing probabilities of heavily used experts.

**Skeptic:** Is this exactly a distance from a uniform histogram? Is 1 its minimum?

**Answer:** No. A perfectly uniform f gives 1, but this bilinear surrogate is not a monotonic imbalance metric. Here is a checked counterexample using our exact top-2 convention:

| Token group | Number of tokens | Router probabilities E0,E1,E2,E3 | Selected |
|---|---:|---|---|
| A | 90 | [0.26, 0.26, 0.24, 0.24] | E0/E1 |
| B | 10 | [0.01, 0.01, 0.49, 0.49] | E2/E3 |

Assignment shares are [0.45,0.45,0.05,0.05]; mean probabilities are [0.235,0.235,0.265,0.265]. MaxVio is 0.8, but the unscaled auxiliary loss is **0.952**, below the balanced value of 1. This does not prove the loss is useless: its training gradients and long-run behavior are different questions. It proves that the loss value alone does not measure histogram balance.

**Our consequence:** Log task loss, actual loads, starvation and runtime separately. Coefficient 0.001 was a chosen setting, not an experimentally established optimum. A small scalar loss contribution also does not guarantee a small gradient contribution.

**23. Why router z-loss? Is it an entropy penalty or a substitute for load balancing? (Section 12; p. 29)**

Z-loss penalizes the squared log-sum-exp of router logits. Load balancing targets expert allocation; these are different targets.

**Skeptic:** If probabilities are unchanged, can z-loss change?

**Answer:** Yes. Add 100 to every logit: softmax probabilities and chosen experts are unchanged, but log-sum-exp increases by 100. Our check leaves probabilities exactly unchanged while squared log-sum-exp increases from about 5.95 to 10,493.99. Therefore z-loss is not simply router entropy, confidence or load imbalance.

**Our evidence:** Z-loss is absent. A stable finite run does not establish that it would never help, and the lesson's successful coefficient elsewhere does not establish that adding it would improve our result.

**24. Why loss-free balancing? Does it leave the language objective untouched? (Section 13; pp. 29–32)**

A controller adds expert-specific biases when choosing IDs, adjusts those biases from recent load and retains original scores for mixture weighting. The [loss-free balancing paper](https://arxiv.org/abs/2408.15664) uses this to avoid an explicit auxiliary-gradient contribution to the task-trained router.

**Skeptic:** Selection changes which functions run. How can that leave learning untouched?

**Answer:** It avoids that extra differentiable loss term, but still changes the executed experts, outputs, gradients and training data received by each expert. 'No auxiliary gradient' is narrower than 'no influence on task learning'. Controller speed, delayed load estimates and oscillation become their own tuning questions.

**Our evidence:** No controller. We cannot claim bias balancing is superior for our run without a controlled comparison to the current auxiliary-loss method.

**25. Why whole-batch rather than microbatch balancing? Can larger scope hide local trouble? (Section 14; pp. 33–35)**

A larger aggregation window lets one subset of data prefer some experts while other subsets compensate. That relaxes pressure for each small group to be uniformly distributed.

**Skeptic:** What if the global histogram is balanced but one microbatch or device receives a huge burst? And if all data is one domain, how does global aggregation guarantee specialists?

**Answer:** Global balance does not guarantee local capacity safety, subgroup fairness or semantic specialization. Conversely, microbatch balancing does not mathematically prohibit all specialization: complementary preferences can coexist within a mixed microbatch. The lesson's strong wording is better read as a possible constraint on specialization, not an impossibility theorem.

**Our evidence:** We balance each 16-sequence microbatch. Accumulating two gradients gives an effective batch of 32 but does not aggregate f and P before computing the balancing objective. Compare scopes at matched update batches, inspecting both global and local load distributions.

**26. Why copy a dense FFN? Doesn't cloning create zero new information? (Section 15; pp. 36–39)**

At conversion, four copies contain redundant representations of the same learned function. The benefit sought is preserving that function while creating independent parameters that can later adapt.

**Skeptic:** If copies are identical, why would the router learn which is best?

**Answer:** Initially, normalized mixtures of identical outputs do not give a meaningful task gradient for relative mixture weights. But different token assignments give copies different parameter gradients; load-balancing gradients can move the router too. Once functions diverge, task weighting can become useful. Copying is an initialization strategy, not an instant quality gain.

**Our evidence:** We checked initial near-equivalence and later weight divergence. The observed benefit arrives after continued training. [Sparse Upcycling](https://arxiv.org/abs/2212.05055) studies reuse of existing dense training investment; its results do not make every clone-based conversion advantageous.

**27. Why not partition or redraw neurons? Does preserving total width preserve the function? (Section 15; pp. 36–39)**

For our two-matrix GELU FFN, partitioning hidden neurons yields several partial contributions. Summing all contributions recovers the original FFN. Selecting only some generally does not.

**Skeptic:** The diagram shows slices adding to a dense width. Doesn't that prove equivalence?

**Answer:** Width equality is parameter/compute accounting, not functional equality. Even when all disjoint slices are present, averaging their outputs instead of summing scales the result. Drop-upcycling deliberately changes some neurons and therefore trades initial fidelity for diversity. Router choices, output scaling and which neurons are retained all matter.

**Our evidence:** We copied full FFNs and normalized mixture weights; no neuron partition or redrawing occurred. A partition or perturbation experiment needs a new equivalence/error check and its own training result.

**28. Why upcycle rather than train from scratch? Is there a universal break-even budget? (Section 15; pp. 36–39)**

Upcycling reuses an existing checkpoint. Its advantage depends on that checkpoint's quality and on the additional compute available.

**Skeptic:** Counting dense pretraining as free makes the method look artificially cheap. And why would a 'twice the budget' rule hold everywhere?

**Answer:** Report both incremental cost when a checkpoint already exists and end-to-end cost when it must be created. There is no universal break-even ratio. [OLMoE's comparison](https://arxiv.org/html/2409.02060v2#S4.SS1.SSS5) found a much earlier scratch-model catch-up than the original upcycling study and discusses differences in model/routing setup.

**Our evidence:** The lineage has 50M dense pretraining targets plus 50M continuation targets, using a second pass over the same corpus. We have no from-scratch MoE control and cannot determine this run's break-even point or claim 100M unique training data.

**29. Why expert parallelism? Why not keep everything local? (Section 16; pp. 40–43)**

Placing different experts on different GPUs can distribute their stored parameters and computation. The cost is sending token vectors and results to the owning devices.

**Skeptic:** If communication costs more than the saved compute, why distribute anything?

**Answer:** Sometimes the model will not fit locally, or a distributed layout improves throughput. Sometimes it is slower. Sparse selection and expert placement are independent: MoE does not logically require multiple GPUs. An ownership map also turns expert-level imbalance into device-level workload, which may or may not be severe depending on placement.

**Our evidence:** EP=1, one GPU, zero inter-GPU expert traffic. The PDF's two all-to-all operations in a forward MoE layer describe a distributed implementation, not a charge paid by our local gather/scatter.

**30. Why keep EP inside one node? Why combine parallelism modes? (Sections 16–17, 19; pp. 40–50)**

Fast local links often make intra-node expert exchanges attractive. Tensor, pipeline and data parallelism divide different dimensions of computation/state and can be combined when capacity or throughput requires it.

**Skeptic:** Is EP equal to GPUs per node a law? Does ZeRO create free memory? Can all communication be hidden behind compute?

**Answer:** No. Network topology, model shape, expert load, batch size and implementation determine the useful layout. Sharding reduces replicated state but adds coordination or communication. Overlap is limited by dependencies, available independent work and resource contention. Ideal bytes/bandwidth calculations are lower-bound-style estimates, not end-to-end measurements. Cross-node EP may be necessary or effective on some systems even if a simple local-link heuristic prefers otherwise.

**Our evidence:** TP=PP=DP=EP=1 and no ZeRO. We cannot claim performance for a distributed layout without measuring it. Splitting expert weights would not automatically divide activations or every other memory component by the same factor.

**31. Why trust the case-study leaderboard or copy its hyperparameters? (Sections 18, 20; pp. 45–47, 51–52)**

Case studies demonstrate available design choices and report outcomes in particular training environments.

**Skeptic:** Different models use different data, tokenizers, tasks, training budgets and software. How can a table identify what caused success?

**Answer:** It usually cannot by itself. Cross-model comparisons suggest hypotheses; controlled ablations are stronger evidence for a mechanism. Compare parameter conventions, active/shared expert accounting, benchmark protocols, training cost and source availability before treating a headline number as comparable. The V5 discussion is a proposed design for a particular project, not a definition of correct MoE.

**Our consequence:** 'A large model uses bias step 0.001' does not establish that 0.001 is optimal here. 'A model freezes its router during RL' does not justify freezing ours during ordinary next-token training. This critique checks selected foundational sources; it is not an independent certification of every recent model claim in the PDF.

**32. Why believe our loss reduction? What does the assignment actually establish? (Section 21; p. 52)**

Held-out cross-entropy measures average next-token predictive performance under a specified tokenizer, target mask and evaluation distribution.

**Skeptic:** Could more compute, a favorable seed, repeated development on this holdout or the data mix explain the gain? Why is generated text still poor?

**Answer:** Those are real limitations. More active compute is a confound in the MoE-vs-dense causal comparison. One training seed does not quantify reproducibility. The holdout was excluded from gradient updates but inspected during development, so it is not a newly untouched final test. Lower average loss can coexist with poor generation, rare-task failure and repetitive decoding.

**Our evidence:** Both arms used matching target exposure/order, and MoE improved all ten held-out source averages. That strengthens the descriptive result. It does not establish statistical significance across training seeds, compute-matched superiority, named expert skills or useful chatbot quality. Our random-routing seeds are inference seeds, not independent training runs.

**Wording to defend:** 'Our full-copy dense-to-MoE conversion continued learning and achieved lower held-out next-token loss than this dense control at equal token exposure, while using more active parameters and more recorded time.'

**Wording not established:** 'MoE is universally better or faster', 'all experts specialized', 'balance is solved', or 'our architecture is optimal'. Also state that the assignment's phrase 'Linear model' was interpreted as the conventional dense Transformer discussed in the lesson, not literally a linear regression model.

**33. Why these exact settings in our assignment? What was evidence-based and what was convenient?**

| Choice | Actual reason/evidence | What has not been demonstrated |
|---|---|---|
| 9 layers, hidden 384, six heads, width 1,664 | Inherited trained dense checkpoint | Optimal shape |
| Four experts per layer | Manageable local experiment | Optimal capacity or expert count |
| Top-2 | Two-path sparse mixture with a task-weighting gradient | Best k |
| Full FFN copies | Reuse trained behavior and allow equivalence checking | Better than partition/drop/scratch |
| MoE in every layer | Consistent conversion of every source FFN | Better than retaining an early dense layer |
| No shared expert | Simple all-routed design | Shared experts would not help |
| FP32 softmax router | Numerically cautious, inexpensive component | Other scoring/precision choices inferior |
| Auxiliary coefficient 0.001 | Chosen balancing setting | Optimal quality/load tradeoff |
| Microbatch scope | Fits the implemented training loop | Equivalent to whole-batch balancing |
| One GPU | Available local hardware | Best distributed throughput |
| Second pass over existing 50M targets | Reuse authorized corpus and exact paired sample order | Benefit of new data or more diverse data |
| Validation loss as principal result | Matches continued-learning objective | Sufficient measure of all language abilities |

A defensible engineering choice can be useful without being optimal. The error is concealing that distinction.

**What would most directly challenge our current conclusions?**

These are proposed experiments, not completed runs:

1. **Capacity/compute alternative:** compare to a dense FFN with active width 3,328, plus a time-matched continuation of the existing dense control. A two-copy dense widening can initially preserve the original FFN by halving the duplicated output contribution, but needs its own initialization and optimizer treatment. Profile actual work; matched parameter arithmetic does not guarantee matched runtime.
2. **Routing necessity:** compare learned routing with fixed-pair or random-routing models trained under those policies. The existing inference perturbations establish dependence on learned routing, not the superiority of its training method.
3. **Balance tradeoff:** vary balance coefficient/scope or introduce a selection-bias controller, keeping the task budget controlled. Report task quality and local/global/device load together; lower MaxVio alone is not success.
4. **First-layer alternative:** retain a dense first layer or use an explicitly shared path under a matched active budget. Test whether concentration reflects a useful common computation or an avoidable training problem.
5. **Expert function:** intervene on experts with output-scale controls and evaluate token categories and rare tasks, not just source averages. Separate immediate ablation effects from recovery after retraining.
6. **Reliability:** repeat promising comparisons across independent training seeds and finalize on a reserved test set not used for choosing the design. Use document/group-level uncertainty estimates rather than treating correlated tokens as independent samples.

The question to carry into every section is: **Compared with which alternative, under which constraint, measured by which outcome, and what result would change our mind?**

**Audit evidence**

`critical_examples.json` records successful numeric checks for FFN permutation equivalence, a non-monotonic balancing-loss example, normalized versus unnormalized top-1 gradients, identical-expert router gradients, and z-loss versus unchanged softmax probabilities. These are small illustrative CPU calculations, not new training runs. All model results above come from the previously saved assignment artifacts. No checkpoint or training dataset was modified.
