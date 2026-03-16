# Coding Standards — ECCV-2026 VLN Project

## Language & Runtime
- Python 3.6 (vlnce env, ETPNav baseline)
- Python 3.8+ (eccv env, new DifNav modules)
- All new code targets Python 3.8+ unless touching ETPNav baseline directly

## Style
- Follow PEP 8; max line length 100
- Use 4-space indentation (no tabs)
- snake_case for variables and functions, CamelCase for classes
- Avoid abbreviations unless domain-standard (e.g., `vp`, `obs`, `traj`)
- Type hints on all new function signatures

## Imports
- Standard library → third-party → local, each group separated by blank line
- Never use wildcard imports (`from x import *`)

## Comments & Docstrings
- Docstrings on all public classes and functions (one-line for trivial, multi-line for complex)
- Inline comments only where logic is non-obvious
- No TODO/FIXME left uncommitted — open a GitHub issue instead

## Configuration
- All hyperparameters via YAML config files or argparse, never hardcoded
- Paths relative to project root (`ETPNav/data/...`), never absolute paths in committed code
- Experiment names must include timestamp: `{exp_name}_{YYYYMMDD_HHMMSS}`

## ML / Training
- Log every 100 steps minimum during training
- First 100 steps: log loss, gradient norm, learning rate, and any domain-specific debug signals
- Use `torch.no_grad()` in all inference/eval loops
- Random seeds: set `torch.manual_seed`, `np.random.seed`, `random.seed` at start of each run
- Save checkpoints every `valid_steps` iterations; keep last 3 + best

## Git
- Commit messages: imperative form, ≤72 chars, e.g. `Add diffusion policy head to ETPNav`
- Never commit: model weights, datasets, .hdf5/.pth/.pt files, result JSONs, training logs
- Always commit: training/eval scripts, config YAMLs, sbatch scripts, source .py files
- Push after every completed task before reporting back

## Sbatch Scripts
- Always specify `--partition=compute`
- Use `nvidia_a100_80gb_pcie` (cost 16) as default GPU; avoid `nvidia_h100` (cost 60)
- Include `source ~/anaconda3/etc/profile.d/conda.sh && conda activate vlnce` in script header
- Redirect stdout/stderr to `logs/<job_name>_%j.out` / `.err`
- Set `HTTPS_PROXY=http://127.0.0.1:17897` if network access needed inside job
