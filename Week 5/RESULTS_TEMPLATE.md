# ERA V5 Session 5 - Results Register

This is the future execution register. Do not enter an observed value until it is backed by a saved configuration, log, checkpoint reference, evaluation artifact, and dataset/tokenizer revision. Planned thresholds remain frozen after execution begins.

## Evidence and accounting validation

| Field | Value |
|---|---|
| Validation type | Week 4 evidence + Session 5 arithmetic systems validation |
| Script | `validate_plan.py` |
| Run status | PASSED |
| Output artifact | `validation_results.json` |
| Model experiment? | No |

Accounting, artifact-consistency, syntax, and CLI smoke checks passed. These checks are not model-training evidence. `validation_results.json` records the individual checks and local-evidence hashes.

## Week 4 measured results

| Metric | Value | Status |
|---|---:|---|
| Raw translation pairs | 5,141,227 | Measured |
| Stage-A surviving pairs | 4,763,996 | Measured |
| Stage-A retention | 92.662627% | Derived |
| Stage-B retained translations | 4,020,129 | Measured |
| Pre-LSH English hubs | 2,977,126 | Measured |
| LSH drops | 4,664 | Measured |
| Final TWM hubs | 2,972,462 | Measured |
| Clean-token estimator output | 215,294,331 | Measured; not frozen-V5 tokens |
| Estimated train tokens | about 210,937,689 | Derived; row-fraction assumption |
| Frozen-V5 train tokens | Pending frozen-V5 retokenization | Local formatted data exists |
| Unique corpus token types | Not measured | No corpus-type inventory was produced |

## Optional Samanantar cleaning-policy micro-proxy

This is a micro-proxy for the Week 4 cleaning decision, not a 1B/3B mixture proxy. Status: **Planned—not executed**.

| Field | Arm A | Arm B | Arm C |
|---|---|---|---|
| Treatment | Schema/non-null/non-empty only | A + Week 4 normalization | B + language/script/alignment/quality/PII/decontamination/dedup |
| Dataset revision | Pending revision resolution | Same resolved revision as A | Same resolved revision as A |
| Model/tokenizer identifier and resolved revision | Assigned at execution | Same pinned revision as A | Same pinned revision as A |
| Parent/cluster leakage | Must be 0 | Must be 0 | Must be 0 |
| Train examples | Assigned at execution; equal-size | Same count as A | Same count as A |
| Optimizer steps | Assigned at execution; fixed | Same count as A | Same count as A |
| Source non-padding tokens | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Target/supervised non-padding tokens | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Combined non-padding token exposure | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Cross-arm combined-token spread | Not available—experiment not executed | Required: <=1% | Required: <=1% |
| Validation cross-entropy | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| chrF | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| sacreBLEU | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Wrong-script output rate | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Source-copy rate | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Empty-output rate | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Repeated-output rate | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Unique target-token yield | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Raw-pair retention | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Wall time | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |
| Peak GPU memory | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed |

### Cleaning-policy gate

| Test: Arm C versus Arm A | Observed | Proposed confirmation | Refutation/revision |
|---|---:|---:|---:|
| chrF | Not available—experiment not executed | >=+1.0 point | <+0.2 point |
| Wrong-script rate | Not available—experiment not executed | >=25% relative reduction | Any increase |
| Source-copy rate | Not available—experiment not executed | >=20% relative reduction | Any increase |
| Validation loss | Not available—experiment not executed | No worse than +2% | Worse than +2% |
| Arm-C raw retention | Not available—experiment not executed | >=70% | <70% |

Run validity also requires an immutable Samanantar revision, identical checkpoint/steps/decoding, zero parent-ID or cluster leakage, equal-size disjoint arm subsets, and a held-out set disjoint by source cluster.

## Planned 1B screening

Overall status: **Planned—not executed**.

### OPUS-starvation selector

| Arm | Planned seeds | Quota attainment | Indic NLL/accuracy | Agentic verified end-state | General guardrail | Decision |
|---|---|---:|---:|---:|---:|---|
| O0 static stratified | 17, 29 | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Pending execution |
| O1 global OPUS/no floors | 17, 29 | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Pending execution |
| O2 protected queues | 17, 29 | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Pending execution |

| O-series gate | Observed | 95% CI | Confirmation | Refutation |
|---|---:|---:|---:|---:|
| Rolling windows meeting both floors | Not available—experiment not executed | Not available—experiment not executed | >=99.0% | <95.0% |
| Final Indic / Agentic share error | Not available—experiment not executed | Not available—experiment not executed | <=0.05 pp / <=0.01 pp | Any confirmation limit exceeded |
| O2-O1 macro Indic | Not available—experiment not executed | Not available—experiment not executed | >=3% relative NLL or >=+2.0 pp accuracy | <1% NLL and <+0.5 pp |
| O2-O1 Agentic verified end-state | Not available—experiment not executed | Not available—experiment not executed | >=+1.5 pp | <+0.5 pp |
| O2-O0 general / English NLL | Not available—experiment not executed | Not available—experiment not executed | No worse than -0.5 pp / +1% relative | General regression >1.0 pp |

### Reasoning-order selector

| Arm | Planned seeds | Effort-order compliance | Length adherence | Accuracy/token AUC | Reasoning accuracy | General+Code | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| R0 flat | 17, 29 | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Pending execution |
| R1 Short to Ultra | 17, 29 | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Pending execution |
| R2 Ultra to Short | 17, 29 | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Pending execution |

| R-series gate | Observed | 95% CI | Confirmation | Refutation |
|---|---:|---:|---:|---:|
| R1-R0 effort compliance | Not available—experiment not executed | Not available—experiment not executed | >=+3.0 pp | No improvement |
| R1-R0 length adherence | Not available—experiment not executed | Not available—experiment not executed | >=+3.0 pp | No improvement |
| R1-R0 accuracy/token AUC | Not available—experiment not executed | Not available—experiment not executed | >=+1.5 pp | <+0.5 pp |
| R1-R0 Long/Ultra accuracy | Not available—experiment not executed | Not available—experiment not executed | >=+1.0 pp | Aggregate reasoning <-0.5 pp |
| R1-R2 accuracy/token AUC | Not available—experiment not executed | Not available—experiment not executed | >=+2.0 pp | R2 exceeds R1 by >=+0.5 pp |
| Short-cap and General+Code guardrail | Not available—experiment not executed | Not available—experiment not executed | No worse than -0.5 pp | Confirmation limit exceeded |

### Mixture, curriculum, and masking screen

| Arm | Run ID | Seed | Tokens | Indic metric | Agentic success@1 | Reasoning | Long-context | General+Code | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1B-A lower control | Assigned at execution | 17; 29 if promoted | 20B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |
| 1B-B Indic isolate | Assigned at execution | 17; 29 if promoted | 20B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |
| 1B-C Agentic isolate | Assigned at execution | 17; 29 if promoted | 20B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |
| 1B-D OPUS dosage | Assigned at execution | 17; 29 if promoted | 20B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |
| 1B-E V5 proposed | Assigned at execution | 17; 29 if promoted | 20B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |
| 1B-F assistant-reasoning ablation | Assigned at execution | 17; 29 if promoted | 20B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |

### 1B gate calculation

| Test | Observed delta | 95% CI | Threshold | Decision |
|---|---:|---:|---:|---|
| Indic `B-A` | Not available—experiment not executed | Not available—experiment not executed | >=+2.0 pp and >=3% relative NLL improvement | Pending execution |
| Indic `D-C` | Not available—experiment not executed | Not available—experiment not executed | >=+2.0 pp and >=3% relative NLL improvement | Pending execution |
| Agentic `C-A` | Not available—experiment not executed | Not available—experiment not executed | >=+1.5 pp success@1 | Pending execution |
| Agentic `D-B` | Not available—experiment not executed | Not available—experiment not executed | >=+1.5 pp success@1 | Pending execution |
| Curriculum reasoning `E-D` | Not available—experiment not executed | Not available—experiment not executed | >=+1.5 pp | Pending execution |
| Curriculum long-context `E-D` | Not available—experiment not executed | Not available—experiment not executed | >=+2.0 pp | Pending execution |
| Assistant-reasoning NLL `E-F` | Not available—experiment not executed | Not available—experiment not executed | >=3% relative improvement | Pending execution |
| Assistant-reasoning Agentic `E-F` | Not available—experiment not executed | Not available—experiment not executed | >=+2.0 pp success@1 | Pending execution |
| General+Code no-harm `E-A` | Not available—experiment not executed | Not available—experiment not executed | >=-0.7 pp | Pending execution |

## Planned 3B confirmation

Overall status: **Planned—not executed**.

| Arm | Run ID | Seed | Tokens | Indic | Agentic | Reasoning | Long-context | General+Code | Status |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 3B-A control | Assigned at execution | 17 | 60B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |
| 3B-A control | Assigned at execution | 29 | 60B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |
| 3B-B locked 1B winner | Assigned at execution | 17 | 60B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |
| 3B-B locked 1B winner | Assigned at execution | 29 | 60B planned | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Not available—experiment not executed | Planned—not executed |

### 3B gate calculation

| Metric | Two-seed mean delta | 95% CI | Confirmation threshold | Refutation threshold | Decision |
|---|---:|---:|---:|---:|---|
| Macro Indic | Not available—experiment not executed | Not available—experiment not executed | >=+1.5 pp | Mean <+0.5 pp | Pending execution |
| Agentic success@1 | Not available—experiment not executed | Not available—experiment not executed | >=+1.0 pp | Mean <+0.5 pp | Pending execution |
| Reasoning composite | Not available—experiment not executed | Not available—experiment not executed | >=+1.0 pp | Mean <+0.5 pp | Pending execution |
| Long-context evidence accuracy | Not available—experiment not executed | Not available—experiment not executed | >=+1.0 pp | Mean <+0.5 pp | Pending execution |
| General+Code | Not available—experiment not executed | Not available—experiment not executed | >=-0.5 pp | Regression >1.0 pp | Pending execution |

## Compute disclosure

The formal 1B and 3B experiments are fully specified but have not been executed. Available RTX 3070 and opportunistic Kaggle resources do not support the proposed multi-arm pretraining study. No model result or promotion decision is claimed. The compact Samanantar cleaning micro-proxy is separately marked **Planned—not executed**.
