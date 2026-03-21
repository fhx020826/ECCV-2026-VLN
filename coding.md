# PAM-FlowNav Coding Standards

## Language & Style

- **Python 3.9**, PyTorch 2.8+, transformers 4.50+
- Follow PEP 8 with max line length **120** characters
- Use **4 spaces** for indentation, no tabs
- Use snake_case for variables/functions, PascalCase for classes
- Type hints for all function signatures in new code
- Remove unused imports and dead code whenever touching a file

## File Organization

```
PAM-FlowNav/
  JanusVLN/
    src/
      evaluation.py         # inference entry point
      dagger.py             # DAgger data collection
      habitat_extensions/   # custom habitat measures
      qwen_vl/
        model/
          modeling_qwen2_5_vl.py  # main model (JanusVLN + VGGT)
          configuration_qwen2_5_vl.py
          loss.py
          vggt/             # VGGT spatial encoder
        data/               # data utils
    config/
      vln_r2r.yaml          # habitat eval config (split=val_unseen)
  scripts/
    slurm/                  # SLURM job scripts
  data/                     # datasets and checkpoints (NOT tracked by git)
  logs/                     # job outputs (NOT tracked by git)
  coding.md                 # this file
  claude.md                 # project analysis
  work.md                   # progress tracking
  user.md                   # user preferences
```

## PyTorch Conventions

- Always use `torch.bfloat16` for inference (model loaded with `torch_dtype=torch.bfloat16`)
- Use `device_map={"": device}` for single-GPU model placement per rank
- Distributed: use `torchrun` with NCCL backend; use `get_rank()`, `get_world_size()` from `utils/dist.py`
- All new tensors: specify `dtype` and `device` explicitly
- Use `with torch.no_grad():` for all inference code
- `model.eval()` must be called after loading

## Model Modification Rules

- `modeling_qwen2_5_vl.py` is the single source of truth for architecture
- New modules added to `Qwen2_5_VLForConditionalGenerationForJanusVLN` class
- Config fields for new modules: add to `configuration_qwen2_5_vl.py` with `getattr(config, "field", default)` for backwards compatibility
- New loss terms: add to `loss.py`, then call from `forward()` in modeling file
- Preserve all existing interfaces; new arguments must have defaults

## SLURM Scripts

- Use absolute paths everywhere (never `${BASH_SOURCE[0]}` on compute nodes)
- Always: `unset ALL_PROXY` at start
- Always: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`
- Always: load cuda module (`module load cuda/12.4.1`)
- GPU selection: prefer `nvidia_a100_80gb_pcie` (cost 16), avoid H100 (cost 60)
- CPU limit: 4 CPUs per GPU (cluster rule → `ntasks-per-node=1`, `cpus-per-task = 4 * num_gpus`)
- If pending > 1 minute on a specific node, cancel and try another node

## Logging & Debugging

- First 100 steps must have verbose debug logs for all known failure modes
- Log format: `[Step {step}] {metric}: {value:.4f}`
- Capture: loss NaN/Inf checks, gradient norms, memory usage, action distribution
- Use `habitat.logger` for habitat-related logs
- Set `MAGNUM_LOG=quiet`, `HABITAT_SIM_LOG=quiet` to suppress simulator noise

## Config Management

- Habitat configs use YAML + OmegaConf; use `with habitat.config.read_write(cfg):` to modify
- Model config changes go in `config.json` (loaded via `AutoConfig.from_pretrained`)
- Do NOT pass model architecture flags as CLI args; use JSON config

## Git Commit Rules

- Commit message: imperative form, ≤72 chars subject line
- Track: source code, configs, SLURM scripts, markdown docs
- Never track: data/, logs/, *.safetensors, *.pt, *.pth, *.bin, result files
- Push at end of every session with proxy enabled

## Proxy on HPC

```bash
# Clash proxy (for git push / pip install)
export HTTPS_PROXY=http://127.0.0.1:18990
export HTTP_PROXY=http://127.0.0.1:18990
export ALL_PROXY=""  # must unset socks proxy (dead)

# For CUDA/pip compilation
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.4.1
```
