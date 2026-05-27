# MoralSteer

Official code for the paper *"MoralSteer: Safety Preserving Fidelity Gains in Morally Complex Role-Playing via Theory-Guided On-Policy Self-Distillation"*.

## Overview

MoralSteer addresses the safety-fidelity tension in LLM-based role-playing: models must stay in character (fidelity) while refusing genuinely harmful requests (safety). We propose **On-Policy Self-Distillation (OPD)** with a moral-complexity-aware curriculum guided by **Moral Foundations Theory (MFT)**, enabling models to navigate the Pareto frontier between safety and fidelity.

## Training

### 1. SFT Warmup

```bash
bash training/scripts/run_sft_warmup.sh /path/to/base_model /path/to/output
```

### 2. OPD Curriculum (3 Stages)

Edit paths in `training/configs/opd_stage{1,2,3}.yaml`, then:

```bash
# Single-stage
python -m training.train_opd --config training/configs/opd_stage1.yaml

# All 3 stages sequentially
bash training/scripts/run_opd_curriculum.sh

# Full pipeline (SFT warmup + 3 OPD stages) for two models
bash training/scripts/train_two_models.sh
```

For multi-GPU FSDP training:

```bash
USE_ACCELERATE=1 NUM_PROCESSES=8 bash training/scripts/train_two_models.sh
```

## Benchmarks

We evaluate on 10 benchmarks across two dimensions:

**Safety** (5 benchmarks): XSTest, AdvBench, BeaverTails, HEx-PHI, SafeDialBench

**Fidelity** (5 benchmarks): RoleBench, CharacterBench, RPEval, Moral Role-Play, HarmBench
