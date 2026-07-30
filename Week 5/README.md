# ERA V5 Session 5 - Mixture and Curriculum Plan

## Decision and accounting basis

This is a testable mixture specification, not a claim that a proxy or production model has been trained. The scale ladder is **1B hypothesis screening -> 3B confirmation -> optional 20B systems pilot -> 240B-parameter production**. Because the production token budget is unapproved, demand is reported per **1.000T selected-token planning block**; for budget `T`, multiply by `T / 1T`.

- Pre-Anneal training uses **98.00%** of selected tokens.
- A separate **2.00% Anneal reserve** is held outside the pre-Anneal mixture.
- Pre-Anneal Always-On floors are **8.00% Indic** and **0.50% Agentic** in every stage.
- Every token has one primary accounting lane. Secondary language, domain, difficulty, context, and effort tags never create extra supply.

Evidence labels are used literally: **Measured** comes from an executed local artifact; **Derived** shows its arithmetic assumption; **Target** is a proposed allocation or gate; **Missing local supply** means no locally admitted dataset; **Pending frozen-V5 retokenization** means local data exists but lacks the final tokenizer count; and **Planned—not executed** means the protocol exists without a training result.

**Session 4 supplies measured cleaning and inventory evidence for the Indic translated tier through AI4Bharat Samanantar. It does not establish verified-native, unverified-native, synthetic, Agentic, Reasoning, or Long-context supply. Those lanes remain explicit acquisition or generation gaps.** It also establishes no local General, Code, Science/Math, or held-back Anneal inventory.

### Current data-gate status

The complete V5 data gate is **not currently met**. Local admitted evidence exists only for Indic translated data. Verified-native Indic, unverified-native Indic, synthetic Indic, General, Code, Science/Math, Agentic, Reasoning, Long-context, and held-back Anneal supply remain local acquisition or generation gaps. This document is a mixture specification and starvation analysis, not authorization to launch production training. Production stays blocked until sources have immutable revisions and license decisions, admitted data has frozen-tokenizer counts and non-overlapping inventories, and the stated 1B and 3B proxy gates pass. No numerical course data-gating threshold is invented.

## Executed Week 4 evidence

The authoritative run is `D:\ERAv5\Week_4\twm_pilot_500k_v1`; older or partial runs are excluded. `WEEK4_AUDIT.md` contains the per-language and filter-level audit.

| Item | Value | Evidence state |
|---|---:|---|
| Source | `ai4bharat/samanantar`, en -> as/bn/gu/hi/kn/ml/mr/or/pa/ta/te | Measured |
| Recorded revision | `main` | Measured; pending immutable revision |
| Recorded license | CC-BY-NC-4.0 / `ADMITTED-NC` | Measured; production license decision still required |
| Raw translation pairs | 5,141,227 | Measured |
| Stage-A surviving pairs | 4,763,996 | Measured |
| Stage-A rejection / retention | 377,231 / 92.662627% | Derived |
| Stage-B retained target memberships | 4,020,129 | Measured |
| Pre-LSH / final TWM hubs | 2,977,126 / 2,972,462 | Measured |
| Near-duplicate hub drops | 4,664 (0.156661%) | Measured |
| Combined formatted-token estimator | 215,294,331 | Measured heuristic; not frozen-V5 tokens |
| Train / validation rows | 2,912,312 / 60,150 | Measured |
| Estimated train tokens | about 210,937,689 | Derived by row fraction; assumes equal average length |
| Exact frozen-V5 tokens | Pending frozen-V5 retokenization | Local formatted data exists |

Raw and Stage-A values are translation-pair units; Stage B counts target-language memberships; TWM hubs aggregate one English source with one or more Indic targets. Therefore `final hubs / raw pairs` is not a retention rate. Samanantar is counted only as **Indic translated**. English source strings are not General-web inventory, and a reversed direction would retain the same parent-pair ID and would not create new source supply.

Source/target token counts, main-run normalization-change counts, exact duplicate-pair counts, final unique aligned-pair counts, and unique token types were not separately recorded. Automatic normalization, LID, alignment, PII, decontamination, and deduplication do not prove native authorship or human verification.

## Exact mixture and evaluation map

| Primary lane | Pre-Anneal share | Total share | Demand per 1T | Candidate datasets (not admitted inventory) | Named evaluations |
|---|---:|---:|---:|---|---|
| General | 53.00% | 51.94% | 519.4B | FineWeb-Edu, DCLM, Dolma, Wikipedia | MMLU-Pro, ARC-Challenge |
| Science/Math | 10.00% | 9.80% | 98.0B | arXiv, PMC OA, licensed textbooks, Stack Exchange | GPQA, MATH |
| Code | 10.00% | 9.80% | 98.0B | The Stack v2/StarCoderData, permissive repos/issues/docs/tests | HumanEval+, LiveCodeBench |
| Reasoning | 10.00% | 9.80% | 98.0B | OpenWebMath, FineMath, NuminaMath, verifier-backed solutions | MATH, GPQA, verified code/science tasks |
| Long-context | 8.50% | 8.33% | 83.3B | Full OA papers, long legal/government documents, repository-scale code | RULER, LongBench v2 |
| Indic | **8.00% floor** | 7.84% | 78.4B | Week 4 Samanantar plus admitted native/translated/synthetic sources | IndicGenBench, IndicXTREME, IN22 |
| Agentic | **0.50% floor** | 0.49% | 4.9B | Replayable tool trajectories and disjoint SWE-style tasks | BFCL, GAIA, held-out SWE-style tasks |
| Anneal reserve | - | 2.00% | 20.0B | Separately held back; composition below | Protected-capability and no-harm gates |
| **Total** | **100.00%** | **100.00%** | **1,000.0B** | | |

Public dataset names are mappings, not inventory claims. Admission requires a resolved revision, license decision, provenance, contamination quarantine, global deduplication, and frozen-V5 token count. Benchmark-targeted selection never trains on the named evaluation items.

### Design decisions and trade-offs

These numbers are starting hypotheses, not measured optima. **General is 53%** because the specialist allocations consume 47%; the residual majority preserves broad language and world knowledge, but the general/no-harm gates may require more. **Code is 10%** to sustain repository-level exposure without allowing code to displace the language base; HumanEval+/LiveCodeBench and the 1B guardrails can reject it. **Science/Math is 10%** as a technical-knowledge anchor distinct from explicit solution traces; GPQA/MATH reveal whether that separation is useful. **Reasoning is 10%** to cover verifier-backed B0-B5 data across all effort tiers; the secondary A-F and R0/R1/R2 studies can revise its treatment.

**Long-context is 8.5%** to supply coherent long documents while bounding sequence cost and instability; RULER/LongBench v2 and context-stability gates can reduce or raise it. **Indic is an 8% protected floor** against global-selector starvation, not a conclusion from Week 4; O0/O1/O2 is its direct test. **Agentic is 0.5%** because executable trajectories are expensive: the floor gives continuous but bounded exposure, and Agentic end-state success plus general-quality guardrails can reject it. **Anneal is 2%** as a small terminal reserve for the highest-quality records; it cannot launch without its unique held-back supply and must survive 3B confirmation.

Inside Indic, **40/20/30/10** gives verified-native data the quality anchor, bounds weaker-provenance native web data, uses translated data for coverage without replacing native use, and caps synthetic data. Only the translated tier has local evidence, so the split remains a hypothesis. The **3x translated replay cap** is a conservative repetition constraint intended to prevent a small parallel corpus from dominating; frozen-tokenizer accounting and the proxy results may revise it.

### Indic tier split

| Indic tier | Share of Indic | Total share | Demand per 1T | Audited local train supply | One-pass coverage |
|---|---:|---:|---:|---:|---:|
| Verified native | 40% | 3.136% | 31.36B | 0 locally admitted tokens | 0% |
| Unverified native | 20% | 1.568% | 15.68B | 0 locally admitted tokens | 0% |
| Translated | 30% | 2.352% | 23.52B | about 0.210937689B | about 0.896844% |
| Synthetic | 10% | 0.784% | 7.84B | 0 locally admitted tokens | 0% |
| **Total** | **100%** | **7.840%** | **78.40B** | **about 0.210937689B** | **about 0.269053%** |

Meeting translated demand from the planning proxy alone would require **111.502x replay**, which is rejected. The **3.0x cap** permits about **0.632813067B**, leaving **22.887186933B** translated acquisition. Total Indic gap is **77.767186933B**: 31.36B verified-native collection, 15.68B unverified-native collection, 22.887186933B translated acquisition, and 7.84B synthetic generation.

### Supply versus demand

Zero below means zero **locally admitted** tokens, not that suitable public data does not exist.

| Lane | Required per 1T | Current local planning supply | Replay cap | Remaining local-evidence gap | Required action |
|---|---:|---:|---:|---:|---|
| General | 519.4B | 0 | 1.0x proposed | 519.4B | Acquire and inventory |
| Science/Math | 98.0B | 0 | 1.0x proposed | 98.0B | Acquire and inventory |
| Code | 98.0B | 0 | 1.0x proposed | 98.0B | Acquire and inventory |
| Reasoning | 98.0B | 0 | 1.0x proposed | 98.0B | Collect/generate with verifiers |
| Long-context | 83.3B | 0 | 1.0x proposed | 83.3B | Preserve coherent structures |
| Indic | 78.4B | about 0.210937689B translated; exact count pending | 3.0x translated | about 77.767186933B | Tier-local acquisition/generation |
| Agentic | 4.9B | 0 | 1.0x proposed | 4.9B | Generate executable trajectories |
| Anneal | 20.0B | 0 | 1.0x proposed | 20.0B | Acquire separately held-back data |

Global parent/near-duplicate clustering occurs before accounting. Anneal and evaluations are split first. Each remaining cluster receives one primary lane by precedence—Agentic, Reasoning, Code, Science/Math, Long-context, Indic, General—so cross-tags never offset two demands. Translated replay is capped per parent cluster, with no repeat inside a 100M-token window.

## Protected dynamic selection

For candidate `i`, OPUS estimates the optimizer update and its first-order benefit on a held-out proxy:

`u_i = -g_proxy^T Δθ_i`, where `Δθ_i = OptimizerUpdate(g_i, optimizer_state)`.

Quality, license, duplication, rarity, and source-diversity checks are gates or penalties, not substitutes for utility. Utilities are normalized within a lane/subqueue and sampled stochastically with Boltzmann probabilities. Proxy batches and named evaluations remain disjoint.

Operational constraints:

1. Protected Indic and Agentic queues receive exactly 8.00% and 0.50% of every pre-Anneal stage and rolling window.
2. A quota ledger tracks selected tokens and debt. Protected queues are serviced before discretionary queues when behind.
3. Indic is balanced by tier/language/source; Agentic by tier/tool family/task subtype. One source family is capped at 20% of its lane.
4. A lane is starved when it cannot meet the next quota without violating admission or replay caps. Training pauses or shortens; OPUS cannot borrow from another lane or lower quality.
5. All Tier-A Agentic and all Anneal records are inaccessible to pre-Anneal scoring or backfill.

The exact 1B selector uses **K=4 candidate microbatches per committed update** with copied optimizer state. Any 240B approximation must achieve **>=0.90 utility-rank correlation** on shadow windows and lose no more than **0.5 pp** on any protected capability in a 1B replay.

## Five-stage curriculum and Anneal

Percentages are of total selected tokens. Each pre-Anneal row preserves the two protected floors.

| Stage | Total | General | Sci/Math | Code | Reasoning | Long | Indic | Agentic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Seed | 5.000% | 2.500% | 0.750% | 0.750% | 0.325% | 0.250% | 0.400% | 0.025% |
| General | 55.000% | 39.000% | 5.000% | 5.000% | 0.750% | 0.575% | 4.400% | 0.275% |
| Reasoning | 20.000% | 6.000% | 2.500% | 2.500% | 7.000% | 0.300% | 1.600% | 0.100% |
| Long-context | 18.000% | 4.440% | 1.550% | 1.550% | 1.725% | 7.205% | 1.440% | 0.090% |
| **Pre-Anneal** | **98.000%** | **51.940%** | **9.800%** | **9.800%** | **9.800%** | **8.330%** | **7.840%** | **0.490%** |
| Anneal | 2.000% | - | - | - | - | - | - | - |

Seed introduces clean B0-B1 and ramps to 4K; General broadens B0-B2 at 4K-8K; Reasoning raises B2-B4 and Medium/Long at 8K-16K; Long-context ramps coherent sequences through 16K, 32K, and then 64K/128K; Anneal lowers learning rate and uses only held-back high-quality data. Linear boundary blends preserve integrated totals. A context increase requires loss rise **<=5%**, gradient-norm p99 below **2x** trailing baseline, and passing throughput/overflow checks.

| Anneal component | Anneal share | Total share | Demand per 1T |
|---|---:|---:|---:|
| Tier-A Agentic | 20% | 0.40% | 4.0B |
| B5 Reasoning | 20% | 0.40% | 4.0B |
| Verified-native Indic | 20% | 0.40% | 4.0B |
| High-quality General/Science | 20% | 0.40% | 4.0B |
| High-quality Code | 10% | 0.20% | 2.0B |
| High-quality Long-context | 10% | 0.20% | 2.0B |
| **Total** | **100%** | **2.00%** | **20.0B** |

Weak data cannot backfill an unavailable Anneal component; the run pauses instead.

## Difficulty and reasoning effort

| Band | Definition | Example |
|---|---|---|
| B0 | Recall, copying, formatting, or one direct step | Extract a date; translate one sentence |
| B1 | One transformation or familiar operation | Apply one formula; issue one valid tool call |
| B2 | Two to three linked steps | Compare two sources and calculate a result |
| B3 | Multi-step composition with distractors | Debug a function from test output |
| B4 | Long dependency chain, planning, or recovery | Recover a failed tool call and finish a multi-file task |
| B5 | Expert, adversarial, sparse-feedback, or long-horizon | Prove/refute a claim or recover through several failures |

Pre-Anneal Reasoning uses **B0 10%, B1 20%, B2 25%, B3 25%, B4 15%, B5 5%**; the Anneal B5 reserve is separate.

| Effort | Supervised assistant-reasoning tokens | Reasoning share | Example |
|---|---:|---:|---|
| Short | 1-256 | 35% | Compute one probability |
| Medium | 257-1,024 | 35% | Diagnose a test and correct the function |
| Long | 1,025-4,096 | 20% | Reconcile sources and derive a recommendation |
| Ultra | 4,097-16,384 | 10% | Execute and recover through a repository-scale task |

Difficulty is labeled from dependencies, distractors, recovery, and uncertainty; effort is the frozen-tokenizer count after masking. They are independent.

## Agentic trajectory and loss contract

Tier A is replayable or equivalently verified, has a valid terminal outcome, provenance, evaluation disjointness, and recovery evidence where applicable. Tier B is replayable without a strong outcome verifier. Tier C is synthetic or weakly grounded. **Tier-A records are physically inaccessible before Anneal**; the pre-Anneal floor may use only admitted Tier-B/C primitives or executable synthetic trajectories.

Supervise only assistant reasoning, tool-call names/arguments, recovery/correction, and final answer. Set `label = -100` on system, user, tool observation/result, environment/execution state, verifier, reward, grader, and outcome-label tokens. Token-origin labels are stored before formatting; the deny-mask overrides serialized role after packing and truncation. Reject a trajectory if any denied-origin token is supervised, a boundary is ambiguous, or no assistant token remains.

## Cleaning continuation

Priority follows the measured starvation: (1) verified-native Indic, (2) unverified-native Indic web, (3) additional translated languages/domains, (4) executable Tier-A Agentic, (5) verifier-backed Reasoning across all effort tiers, (6) coherent Long-context sources, and (7) licensed Code, Science/Math, and General inventories. Synthetic Indic remains a generation gap and cannot replace native acquisition.

### Next executable cleaning gate

**Samanantar frozen-tokenizer inventory gate — Planned—not executed.**

Produce `samanantar_frozen_token_inventory.json` containing the immutable Samanantar revision, frozen candidate-tokenizer revision, per-language source tokens, target tokens, combined formatted tokens, exact train/validation tokens, parent-pair and hub counts, and content hashes.

Accept only if all 11 language directions are present; totals reconcile with the authoritative split; no mutable `main` remains; reversed examples do not increase source-supply counts; train and validation stay parent/hub disjoint; and the README supply table is regenerated from the artifact.

## Optional Samanantar cleaning micro-proxy

Status: **Planned—not executed**. This is a compact cleaning-policy test, not a 1B/3B mixture proxy.

| Arm | Treatment |
|---|---|
| A | Schema/non-null/non-empty checks only |
| B | A plus Week 4 normalization |
| C | B plus script/language/alignment/quality/PII/decontamination/dedup |

Use an immutable Samanantar revision and pinned model/tokenizer revision. Arms are parent/cluster-disjoint, share one clean held-out set, and use identical optimizer steps, batch construction, checkpoint, and decoding. Before training, the script tokenizes all candidates and deterministically selects the scheduled examples so combined non-padding source-plus-target exposure differs by **<=1%** across arms; it records source, target, supervised, and combined counts.

Metrics are validation cross-entropy, chrF, sacreBLEU, wrong-script, source-copy, empty/repeated output, unique target-token yield, retention, wall time, and peak memory. Confirm C over A with **>=+1.0 chrF**, **>=25% relative wrong-script reduction**, **>=20% relative source-copy reduction**, loss no worse than **+2%**, and **>=70% retention**. Refute or revise if chrF gain is **<+0.2**, either validity rate worsens, loss worsens by **>2%**, or retention is **<70%**.

## Primary formal proxy and confirmation

Status: **Planned—not executed**. The primary hypothesis is:

> Protected Indic and Agentic queues should prevent capability starvation under dynamic selection without materially harming General and Code performance.

### Primary 1B proxy: OPUS starvation

Train each 1B arm for **20B selected tokens**, using **seed 17** and **seed 29** for promotion.

| Arm | Selector treatment |
|---|---|
| O0 | Static stratified sampling with proposed lane quotas |
| O1 | English-heavy global OPUS, one queue, no floors |
| O2 | Lane-local OPUS with protected 8.00% Indic and 0.50% Agentic queues |

Hold fixed the timestamped candidate stream/IDs, tokenizer, initialization family, optimizer, batch/context schedule, proxy batches, evaluation sets, total eligible supply, and all non-selector settings.

Measure rolling and aggregate quota attainment; macro Indic byte-normalized NLL/accuracy; Agentic verified end-state success; MMLU-Pro/ARC and General+Code guardrails; selector throughput; and instability count.

**Confirm O2** if at least **99.0%** of rolling windows meet both floors; final shares are within **+/-0.05 pp Indic** and **+/-0.01 pp Agentic**; versus O1, Indic improves by **>=3% relative NLL** or **>=+2.0 pp accuracy** and Agentic success improves by **>=+1.5 pp**; and versus O0, General is no worse by **0.5 pp** and English NLL by **1% relative**. Both seeds must agree in sign, and paired item-bootstrap 95% intervals must exclude zero for primary gains; these intervals do not estimate training-seed variance.

**Refute** if quota attainment is **<95%**, either capability gain is **<+0.5 pp** with **<1% NLL improvement**, or a guardrail regresses by **>1.0 pp**. Promote only after both-seed confirmation and selector-throughput review.

### Primary 3B confirmation

Train two 3B arms for **60B selected tokens each**, using **seeds 17 and 29** with fixed tokenizer, initialization family, optimizer, batch/context schedule, evaluations, decoding, and data order:

| Arm | Recipe |
|---|---|
| 3B-A control | 57.25% General, 10% Science/Math, 10% Code, 10% Reasoning, 8.5% Long, 4% Indic, 0.25% Agentic; flat schedule; required mask |
| 3B-B locked winner | Recipe hash containing only selector, mixture, curriculum, reasoning-order, and masking choices that passed their 1B gates |

The 3B run introduces no new treatment. Confirm if the two-seed mean improves macro Indic by **>=+1.5 pp**, Agentic success by **>=+1.0 pp**, Reasoning by **>=+1.0 pp**, and Long-context evidence accuracy by **>=+1.0 pp**, while General+Code is no worse than **-0.5 pp**, with paired item-bootstrap intervals excluding zero for gains. Refute if either seed reverses a gain by **>0.5 pp**, any mean intended gain is **<+0.5 pp**, General+Code regresses by **>1.0 pp**, or training is unstable.

The detailed R0/R1/R2 reasoning-order test and 1B-A through 1B-F mixture/curriculum/masking screen are preserved in `ADDITIONAL_EXPERIMENTS.md` and run only if resources permit. `RESULTS_TEMPLATE.md` remains the future execution register.

The formal 1B and 3B experiments are fully specified but have not been executed. Available RTX 3070 and opportunistic Kaggle resources do not support the proposed multi-arm pretraining study. No model result or promotion decision is claimed.

## Research basis and alternatives considered

The plan uses research to choose what must be tested, not to manufacture certainty about the answer. OPUS supplies the candidate-level selection objective; dynamic-mixture work motivates revisiting allocation as model state changes; capacity-aware work motivates a larger-model confirmation; curriculum studies motivate matched-compute ordering controls; and multilingual studies motivate language-level sufficiency and quality audits. V5 adds hard capability floors, admission rules, source caps, staged budgets, and promotion gates because unconstrained utility selection does not by itself guarantee coverage, provenance, or production safety.

| Design choice | Research basis | What the paper does not prove |
|---|---|---|
| Projected dynamic selection | [OPUS, 2602.05400v2](https://arxiv.org/abs/2602.05400v2) motivates optimizer-induced projected utility, scalable approximation, and stochastic per-iteration selection. | It does not propose V5 floors, queues, stages, or percentages. |
| Changing mixture over training | [TiKMiX, 2508.17677v1](https://arxiv.org/abs/2508.17677v1) shows that domain preferences can vary with progress and scale and motivates periodic influence-aware adjustment. | Its domain-level Group Influence is not OPUS candidate utility and does not validate the V5 mixture. |
| Scale confirmation | [CAMEL, 2603.08022v2](https://arxiv.org/abs/2603.08022v2) models capacity/mixture interaction and motivates checking effect retention across model sizes. | It does not guarantee that a 1B winner transfers to 3B or 240B. |
| Curriculum controls | [Curriculum dynamics, 2601.21698v2](https://arxiv.org/abs/2601.21698v2) motivates matched-compute ordering and stability analysis; [Beyond Random Sampling, 2506.11300v2](https://arxiv.org/abs/2506.11300v2) motivates warmup, pacing, and auxiliary difficulty signals. | Neither tested V5's exact stages or R0/R1/R2 ordering. |
| Multilingual sufficiency and quality | [Multilingual mixtures, 2510.25947v1](https://arxiv.org/abs/2510.25947v1) motivates language-level token sufficiency and both-scale validation; [cross-lingual quality classifiers, 2604.20549v1](https://arxiv.org/abs/2604.20549v1) motivates calibrated per-language quality audits. | Neither proves 8% Indic, native authorship, or human verification. |

**Scientific limitations.** The cited literature motivates dynamic selection, constrained allocation, scale-aware validation, curriculum controls, and multilingual quality auditing. It does not establish that the proposed 8% Indic floor, 0.5% Agentic floor, 2% Anneal reserve, Indic tier split, 3x replay cap, or exact stage shares are optimal. These remain pre-registered V5 hypotheses requiring 1B screening and 3B confirmation. Citation does not convert a candidate dataset into admitted inventory; published results do not replace local supply accounting; benchmark-targeted selection requires strict contamination quarantine; and no citation is evidence that an unexecuted V5 proxy passed.

## Promotion gates

Production remains blocked until every source has a resolved revision, license decision, and content hash; admitted shards are frozen-V5 retokenized; non-overlapping supply tables are regenerated; post-packing contamination and Agentic mask audits pass; Anneal remains held back and reconciled; primary 1B and 3B gates pass or the plan is revised; and any production OPUS approximation passes its exact-utility and protected-capability gates.

Accounting, artifact-consistency, syntax, and CLI smoke checks passed. These checks are not model-training evidence.
