# ERA V5 Session 5 - Additional Experiments

These are secondary studies to run only if resources permit. They are not the primary small-proxy study. Status: **Planned—not executed**. No observed result, confidence interval, model result, or promotion decision is claimed.

All public dataset quantities remain provisional until retokenized with the frozen V5 tokenizer. Every formal arm retains the Agentic token-origin deny-mask described in `README.md`.

## 1B reasoning-order experiment

Each arm trains a 1B-parameter model for **20B selected tokens** using the same fixed controls and seeds **17 and 29**. Every arm reuses the same reasoning candidate IDs and frozen-tokenizer count in each tier, keeps the non-reasoning stream in the same order, and receives exactly the same total Short, Medium, Long, and Ultra reasoning tokens; only reasoning-tier order changes.

| Arm | Reasoning-tier order |
|---|---|
| R0 | Flat interleaving of the fixed Short/Medium/Long/Ultra counts |
| R1 | Short -> Medium -> Long -> Ultra |
| R2 | Ultra -> Long -> Medium -> Short |

Primary metrics are requested-effort order compliance, output-length adherence to the four tier ranges, accuracy-versus-token-budget area under the curve at 256/1,024/4,096/16,384-token caps, reasoning accuracy, and general+code guardrails.

**Confirm R1** if, versus R0, it improves effort-order compliance by **>=+3.0 pp**, length adherence by **>=+3.0 pp**, accuracy-versus-token-budget AUC by **>=+1.5 pp**, and Long/Ultra accuracy by **>=+1.0 pp**; it must also exceed R2 AUC by **>=+2.0 pp** while losing no more than **0.5 pp** at the Short cap or on general+code. Both seeds must agree in sign and paired-bootstrap 95% intervals over evaluation items must exclude zero for primary gains.

**Refute the ascending-order hypothesis** if its AUC gain over R0 is below **+0.5 pp**, either compliance metric fails to improve, aggregate reasoning accuracy falls by more than **0.5 pp**, or R2 exceeds R1 AUC by **>=+0.5 pp**.

## 1B mixture, curriculum, and masking screen

Each arm trains a 1B-parameter model for **20B selected tokens** with the same frozen tokenizer, initialization family, optimizer, batch/context schedule, data-order construction, evaluation harness, and non-varied lane ratios. Differences removed from Indic/Agentic are reallocated only to General. Every arm first runs **seed 17**; any arm comparison proposed for promotion is repeated with **seed 29** (mandatory also when a primary result is within 0.5 percentage points of a gate).

| Arm | General pre-share | Indic pre-share | Agentic pre-share | Curriculum | Agentic loss |
|---|---:|---:|---:|---|---|
| 1B-A lower control | 57.25% | 4.00% | 0.25% | Flat | Required mask |
| 1B-B Indic isolate | 53.25% | 8.00% | 0.25% | Flat | Required mask |
| 1B-C Agentic isolate | 57.00% | 4.00% | 0.50% | Flat | Required mask |
| 1B-D OPUS dosage | 53.00% | 8.00% | 0.50% | Flat | Required mask |
| 1B-E V5 proposed | 53.00% | 8.00% | 0.50% | Five stages | Required mask |
| 1B-F assistant-reasoning ablation | 53.00% | 8.00% | 0.50% | Five stages | Required deny-mask; additionally mask assistant reasoning |

Numerical hypotheses:

- **Indic dosage:** `B-A` and `D-C` each improve macro Indic held-out accuracy by at least **+2.0 pp** and macro byte-normalized NLL by at least **3% relative**.
- **Agentic dosage:** `C-A` and `D-B` each improve held-out task success@1 by at least **+1.5 pp**, with tool-call validity no worse by more than **0.5 pp**.
- **Curriculum:** `E-D` improves reasoning composite by at least **+1.5 pp** and long-context evidence-grounded accuracy by at least **+2.0 pp**, while short-context accuracy falls by no more than **0.5 pp**.
- **Assistant-reasoning supervision:** `E-F` improves supervised-assistant NLL by at least **3% relative** and Agentic success@1 by at least **+2.0 pp**, with tool-call validity non-inferior within **0.5 pp**. Both arms retain the forbidden-origin deny-mask.
- **No harm:** `E-A` is no worse than **-0.7 pp** on general+code and no worse than **+1% relative NLL** on English validation.

**Confirmation threshold:** all four intended effects and no-harm pass; paired-bootstrap 95% confidence intervals exclude zero for each primary gain.

**Refutation threshold:** an isolated gain is below **+0.5 pp**, its NLL worsens by more than **1%**, any no-harm limit is exceeded, or assistant-reasoning supervision produces no assistant-token improvement.

Mandatory metrics: macro Indic accuracy, byte-normalized NLL, Agentic success@1, valid tool-call rate, recovery success, reasoning composite, long-context evidence accuracy, general+code composite, English NLL, tokens/s, peak VRAM, and instability count.

