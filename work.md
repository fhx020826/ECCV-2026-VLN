# PAM-FlowNav Work Progress

## Project Overview

PAM-FlowNav (Progress-Aware Memory-Conditioned Flow Policy) for ECCV 2026.
Host codebase: JanusVLN (ICLR 2026).
Work dir: `/home/hxfeng/PAM-FlowNav/`
Conda env: `pamflownav` (Python 3.9)

---

## Phase 1: Environment & Baseline Reproduction

### Status: RUNNING (eval job queued)

**Goal**: Reproduce JanusVLN_Extra val_unseen results within ±1% of official numbers.

### Completed Setup Steps

| Step | Status | Notes |
|------|--------|-------|
| Clone JanusVLN | DONE | `JanusVLN/` from github.com/MIV-XJTU/JanusVLN |
| conda env pamflownav | DONE | Python 3.9 |
| habitat-sim 0.2.4 | DONE | conda install |
| habitat-lab 0.2.4 | DONE | editable install from `habitat-lab/` |
| torch 2.8.0+cu128 | DONE | upgraded from 2.5.1 |
| flash-attn 2.8.3 | DONE | compiled from source (~60min), CUDA 12.4 + sm_80 |
| transformers 4.50.0 | DONE | |
| JanusVLN_Extra download | DONE | 17.6GB, 4 safetensors, VGGT embedded |
| JanusVLN_Base download | DONE | 18GB, 4 safetensors |
| Qwen2.5-VL-7B-Instruct | DONE | symlink configured |
| MP3D 11 val_unseen scans | DONE | symlink to ETPNav data |
| R2RVLN-v1 val_unseen | DONE | 1839 episodes |
| All import checks | DONE | All pass on login node |
| Data path validation | DONE | All paths resolve correctly |
| SLURM eval script | DONE | `scripts/slurm/eval_janusvln_val_unseen.sbatch` |
| git repo init | DONE | PAM-FlowNav root, main branch |

### Current Eval Job

- **Job ID**: 288 (submitted 2026-03-21)
- **Node**: gpu18 (4× nvidia_a100_80gb_pcie)
- **Estimated start**: 2026-03-22 09:19 AM (waiting for gpu18 to free up)
- **Script**: `scripts/slurm/eval_janusvln_val_unseen.sbatch`
- **Checkpoint**: `data/checkpoints/misstl/JanusVLN_Extra`
- **Output**: `logs/eval_extra_unseen_{timestamp}/result.json`
- **Note**: Job 271 (1×H100) was cancelled — too slow (~47s/episode × 1839 = 24h, exceeds 8h limit)

### SLURM Config (corrected)

```bash
#SBATCH --ntasks-per-node=1      # 1 SLURM task (torchrun forks 4 processes)
#SBATCH --cpus-per-task=16       # 4 CPUs/GPU × 4 GPUs = 16 (cluster limit)
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:4
```

### Expected Eval Metrics (±1% tolerance)

| Metric | JanusVLN_Extra Official |
|--------|------------------------|
| SR (val_unseen) | ~72% |
| SPL | ~62% |
| NE | ~3.2m |
| OS | ~82% |

---

## Phase 2: PAM-FlowNav Core Innovations

### Status: PLANNING

**Goal**: Add rectified-flow action head + progress/affordance auxiliary supervision.

#### Planned Components

1. **Flow Action Head** (`src/qwen_vl/model/flow_head.py`)
   - Replace discrete action tokens with continuous waypoint generation
   - Rectified flow matching conditioned on LLM hidden states

2. **Progress Head** (`src/qwen_vl/model/progress_head.py`)
   - Predict navigation progress [0,1] from LLM hidden state
   - Auxiliary loss during DAgger fine-tuning

3. **Affordance Head** (`src/qwen_vl/model/affordance_head.py`)
   - Candidate passability prediction using VGGT 3D features
   - Guides action selection at inference time

#### Config Changes Required

```json
{
  "use_flow_head": true,
  "flow_steps": 10,
  "use_progress_head": true,
  "use_affordance_head": true
}
```

---

## Phase 3: Fine-tuning

### Status: NOT STARTED

- DAgger data collection via `src/dagger.py`
- Training on R2R-CE train split (requires full 90 MP3D scans, currently only 11)
- Need to download remaining MP3D training scans

---

## Known Issues Log

| Date | Issue | Fix |
|------|-------|-----|
| 2026-03-20 | flash-attn: CUDA_HOME not set | Load cuda/12.4.1 module |
| 2026-03-20 | flash-attn: GitHub download blocked | Use proxy http://127.0.0.1:18990 |
| 2026-03-21 | SLURM CPU limit: 32 > 16 | Change to ntasks-per-node=1, cpus-per-task=16 |
| 2026-03-21 | All A100 nodes full after maintenance | Job 288 queued for tomorrow (gpu18) |
| 2026-03-21 | Triton 0 active drivers on H100 | 3-step fix: prebuilt flash-attn wheel + remove cuda/12.4.1 module + delete stale triton-3.1.0.dist-info |
| 2026-03-21 | gpu05 shows 7 free GPUs but only 2 CfgTRES slots | scir-watch shows physical GPUs, not SLURM-schedulable TRES |

---

## File Cleanup Candidates

- `logs/flash_attn_install.log` — flash-attn install complete, can delete
- `logs/eval_janusvln_unseen_242.out/.err` — will be created; keep until results verified
