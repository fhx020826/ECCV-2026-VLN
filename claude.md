# PAM-FlowNav: Project Analysis (claude.md)

## Research Goal

PAM-FlowNav (Progress-Aware Memory-Conditioned Flow Policy) extends JanusVLN by adding:
1. **Rectified-flow action head** — continuous action generation via flow matching
2. **Progress auxiliary head** — predicts navigation progress [0,1] along trajectory
3. **Affordance auxiliary head** — binary passability prediction for candidate viewpoints

Host codebase: JanusVLN (ICLR 2026), Alibaba MIV-XJTU

---

## JanusVLN Architecture

### Overview

JanusVLN uses a **dual implicit memory** architecture:
- **Semantic KV cache** (`StartRecentKVCache`): LLM key-value pairs, `start_size=8, recent_size=48`
- **Spatial KV cache** (`past_key_values_vggt`): VGGT 3D features from past observations

Two encoders fused through `VGGTMerger`:
1. **Qwen2.5-VL-7B** (semantic backbone): visual encoder + 7B LLM
2. **VGGT-1B** (spatial encoder): processes current frame → 3D spatial features

### Key Files

| File | Role |
|------|------|
| `src/evaluation.py` | Entry point for distributed eval; `JanusVLN_Inference`, `VLNEvaluator` |
| `src/dagger.py` | DAgger data collection for fine-tuning |
| `src/qwen_vl/model/modeling_qwen2_5_vl.py` | Main model class (2334 lines) |
| `src/qwen_vl/model/configuration_qwen2_5_vl.py` | Config with `use_vggt_feature`, `add_ground_classifier`, etc. |
| `src/qwen_vl/model/loss.py` | Loss utilities: `normalize_pointcloud`, `check_and_fix_inf_nan` |
| `src/qwen_vl/model/vggt/` | VGGT spatial encoder (frozen during VLN eval) |
| `src/habitat_extensions/` | Custom habitat measures: `DistanceToGoal`, etc. |
| `src/utils/dist.py` | NCCL distributed init via RANK/WORLD_SIZE/LOCAL_RANK env vars |
| `config/vln_r2r.yaml` | Habitat config: RGB+depth 640×480, hfov=79, step=0.25m, turn=15° |

### Model Forward Pass (Evaluation)

```
Observation (RGB + Depth 640×480, hfov=79°)
    │
    ├─→ Qwen2.5-VL ViT encoder → visual tokens
    │        └─→ Semantic KV Cache (start=8, recent=48) ──────────┐
    │                                                              │
    └─→ VGGT encoder (frozen) → 3D spatial features              │
             └─→ VGGTMerger → spatial tokens                      │
                      └─→ past_key_values_vggt (spatial KV) ───────┤
                                                                   │
                              LLM (Qwen2.5-7B) ←────────────────────┘
                                    │
                              Action token generation
                         (MOVE_FORWARD / TURN_LEFT / TURN_RIGHT / STOP)
```

### Inference Details

- `JanusVLN_Inference.call_model()`: builds multimodal messages, uses `processor.apply_chat_template`
- Max history: `num_history=8` past frames (controlled in `VLNEvaluator`)
- Generation: greedy decode (`temperature=0, num_beams=1, max_new_tokens=24`)
- Action parsing: string match on generated tokens for 4 discrete actions
- `images_vggt`: only LAST frame of the history window is passed to VGGT

### Config Parameters (config.json)

```json
{
  "architectures": ["Qwen2_5_VLForConditionalGenerationForJanusVLN"],
  "use_vggt_feature": true,
  "reference_frame": "first",
  "lam": 0.2,
  "add_ground_classifier": false,
  "vggt_merger_hidden_dim": 4096
}
```

### Distributed Evaluation

- `torchrun --nproc_per_node=4` launches 4 processes (one per GPU)
- Each process loads full model to its GPU (`device_map={"": device}`)
- Episodes partitioned across ranks; results gathered and averaged
- Output: `logs/eval_extra_unseen_{timestamp}/result.json` (JSONL, one line per episode)
- Metrics: SR (Success Rate), SPL, OS (Oracle Success), NE (Navigation Error)

---

## Environment Setup

### conda env: `pamflownav` (Python 3.9)

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate pamflownav
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

Key packages:
- `torch==2.8.0+cu128` (upgraded from 2.5.1)
- `flash-attn==2.8.3` (compiled from source, CUDA 12.4, sm_80)
- `transformers==4.50.0`
- `habitat-sim==0.2.4`, `habitat-lab==0.2.4` (editable: `habitat-lab/`)
- `qwen-vl-utils`, `openai-clip`, `omegaconf`

### Data Paths (all relative to `JanusVLN/`)

```
data/                             → symlink to /home/hxfeng/PAM-FlowNav/data/
data/scene_datasets/mp3d/         → 11 val_unseen .glb scenes (symlink)
data/datasets/r2r/val_unseen/     → R2RVLN-v1, 1839 episodes (symlink)
data/checkpoints/misstl/
  JanusVLN_Extra/                 → 17.6GB, 4 safetensors (VGGT embedded)
  JanusVLN_Base/                  → 18GB, 4 safetensors
```

### Qwen2.5-VL-7B Symlink

```
data/checkpoints/Qwen/Qwen2.5-VL-7B-Instruct
  → ../Qwen2___5-VL-7B-Instruct
```

---

## Official JanusVLN Results (R2R-CE val_unseen)

| Model | SR | SPL | NE | OS |
|-------|-----|-----|-----|-----|
| JanusVLN_Extra | ~72% | ~62% | ~3.2m | ~82% |
| JanusVLN_Base | ~69% | ~59% | ~3.5m | ~80% |

(Exact numbers from paper; Phase 1 goal: reproduce within ±1%)

---

## Phase 2: PAM-FlowNav Modifications

### Planned Changes to `modeling_qwen2_5_vl.py`

1. **Flow Action Head** (`src/qwen_vl/model/flow_head.py`):
   - Rectified flow matching on continuous waypoint coordinates
   - Conditioning on LLM hidden states + spatial KV cache features

2. **Progress Head** (`src/qwen_vl/model/progress_head.py`):
   - MLP: LLM hidden state → scalar [0, 1]
   - Supervised by ratio of steps completed vs total path length

3. **Affordance Head** (`src/qwen_vl/model/affordance_head.py`):
   - Per-candidate binary classifier: passable vs blocked
   - Uses VGGT features + LLM context

### New Config Fields (to add)

```json
{
  "use_flow_head": true,
  "flow_steps": 10,
  "use_progress_head": true,
  "use_affordance_head": true
}
```

---

## Known Issues & Workarounds

1. `ALL_PROXY` dead proxy → always `unset ALL_PROXY` in SLURM scripts
2. `LD_LIBRARY_PATH` not set by conda → add `$CONDA_PREFIX/lib` explicitly
3. SLURM CPU limit: 4 CPUs/GPU → use `ntasks-per-node=1, cpus-per-task=16` for 4-GPU jobs
4. flash-attn is hardcoded in evaluation.py (line 259) → must install, cannot skip
5. `JanusVLN/data` must be symlinked to `PAM-FlowNav/data` for relative config paths to work
6. VGGT weights are embedded in JanusVLN_Extra safetensors — no separate download needed
