# Governance-Aware Multi-Agent LLM Architectures for Regulated Knowledge-Intensive Tasks

Code and results for my MSc dissertation in Data Science and Machine Learning at UCL (September 2026).

**Supervisors:** Prof. Philip Treleaven (UCL Computer Science) · Dr Omer Gunes (Oxtractor Ltd./ Oxford University Computer Science)

---

## Overview

AI agents are increasingly used in regulated, high-stakes domains such as finance and law. In these settings, a correct answer is not enough: decisions must be traceable, errors must be attributable to a specific component, and humans must keep control over the system.

This project treats **traceability, error attribution and controllability as first-class architectural constraints** rather than post-hoc compliance additions. It proposes a modular multi-agent system with three governance-relevant layers, each studied as a design variable:

1. **A Planner agent**, whose reasoning can be explicit, latent or soft-prompt based
2. **A structured agent-to-agent (A2A) communication protocol**, with configurable topology and message schema
3. **A verifier-mediated governance layer**, using an LLM judge to audit the system

The central claim: **governance is not a single quantity.** Information preservation, outcome correctness, process integrity and post-hoc trace auditability are separable properties, and an architecture can be strong on one while weak on the others.

---

## Research questions

### RQ1 — Planner reasoning strategy
Compares three ways for a planner to reason, each reproduced under its own published backbone and benchmark:

| Strategy | Mechanism | Backbone | Benchmark |
|---|---|---|---|
| **S1** | Explicit chain-of-thought with test-time budget forcing | Qwen2.5-32B-Instruct (fine-tuned, 4-bit) | MATH500, AIME24, GPQA Diamond |
| **Coconut** | Continuous latent-space reasoning | GPT-2 | GSM8K |
| **SoftCoT** | Frozen-backbone soft-prompt reasoning | Qwen2.5-7B-Instruct | GSM8K |

Because backbones and benchmarks differ, strategies are compared through behavioural signatures rather than ranked on accuracy. Five scale- and task-invariant metrics: normalised accuracy gain, overthinking ratio, trace coverage, faithfulness under perturbation, and an auditability composite.

### RQ2 — Communication topology and message schema
A **3 × 2 factorial experiment** on [AgentCollabBench](https://github.com/aritra741/AgentCollabBench) (204 runs, GPT-4.1-mini):

- **Topology:** hierarchical · decentralised · shared-pool
- **Message schema:** untyped free text · typed JSON-schema-constrained output

Metrics: information preservation (Radioactive Tracer Durability), coordination latency (raw and normalised), and turn-length coverage.

### RQ3 — Governance evaluation framework
Two verifier instruments built on an LLM judge (Claude Sonnet 5):

- **V_live (process verifier):** replayed turn by turn over the 204-run RQ2 corpus, checking whether the declared message structure holds and whether a turn contradicts an earlier claim
- **V_post (post-hoc trace verifier):** judges complete reasoning traces against injected-perturbation ground truth from RQ1

---

## Key findings

- **Topology is the dominant structural factor** for information preservation (η² ≈ .27) and for turn-length coverage (η² ≈ .09), with the two effects running in opposite directions.
- **Typed messaging** leaves information preservation essentially unchanged but reduces inter-agent self-contradiction roughly fourfold.
- **The post-hoc verifier is precise but not exhaustive:** precision of 0.968, but it recovers an injected fault in 84% of SoftCoT traces against 36% of S1 traces, two strategies equivalent in terms of trace emission.
- **Producing a trace is not the same as being auditable.** An auditability requirement expressed as "the system must produce a trace" is not equivalent to one expressed as "errors must be measurably recoverable from that trace".

The thesis also contributes a labelled corpus of perturbation-injected reasoning traces, used as ground truth for V_post.

---

## Repository structure

```
├── changelog/          # Day-by-day patch notes and experiment log
├── pilot_results/      # Pilot runs used to validate the pipeline before full-scale execution
├── full_run_results/   # Full RQ2 run outputs, including normalised latency analysis
├── rq1/                # RQ1: planner reasoning strategies (S1, Coconut, SoftCoT), metrics and corrections
├── rq2_tasks/          # RQ2: AgentCollabBench tasks and topology × schema configurations
├── rq3/                # RQ3: V_live replay pipeline and V_post trace verifier
└── scripts/            # Shared analysis scripts (statistical tests, multiple-testing correction)
```

---

## Setup

**Infrastructure.** RQ1 reproductions ran on a single AWS EC2 `g6e.xlarge` instance (one NVIDIA L40S, 46 GB VRAM, Ubuntu 24.04). RQ2 and RQ3 call models through the OpenAI and Anthropic APIs.

**Models**
- RQ1: Qwen2.5-32B-Instruct (S1), GPT-2 (Coconut), Qwen2.5-7B-Instruct (SoftCoT)
- RQ2: GPT-4.1-mini, locked through a dedicated provider configuration
- RQ3: Claude Sonnet 5 as judge

**API keys.** Set your keys as environment variables before running RQ2 or RQ3:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
```

**Logging.** Runs are logged as append-only JSONL, one record per instance, and are resumable.

---

## Limitations

- RQ1 strategies run on different backbones and benchmarks, so accuracy is not comparable across them.
- Single seed and a single benchmark per strategy in several RQ1 conditions.
- The judge in RQ3 samples at non-zero temperature, so re-execution may not reproduce identical verdicts.
- Experiments use general reasoning and collaboration benchmarks, not a regulated finance or legal corpus. Evaluation on a regulated corpus is the main direction for future work.

---

## Citation

```bibtex
@mastersthesis{letayf2026governance,
  title  = {Governance-Aware Multi-Agent LLM Architectures for Regulated Knowledge-Intensive Tasks},
  author = {Letayf, Manon},
  school = {University College London},
  year   = {2026},
  type   = {MSc Dissertation}
}
```

## Contact

Manon Letayf · [LinkedIn](#) · [email](#)
