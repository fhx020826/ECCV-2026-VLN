# User Preferences & Important Commands

## User Preferences

- No emojis in responses
- Concise, direct communication (no preamble)
- Chinese responses when user writes in Chinese
- Code comments in English
- Max line length: 120 chars

## Workflow Rules

1. Delete unused/obsolete code when updating — keep codebase clean
2. Git commit after every meaningful code change; push at end of every session
3. Long jobs: submit via SLURM, confirm running + first 100 steps OK, then report
4. Monitor training first 100 steps for anomalies; kill and fix if severe
5. Never submit job requesting H100 unless all other GPUs occupied
6. If job pending >1 min on a specific node → cancel and try different node
7. Run pre-flight checks (syntax, logic, import) before every job submission
8. Update work.md/claude.md after every session; keep user.md for persistent prefs
9. Notify user before deleting any files; list cleanup candidates at session end
10. All new code goes to `/home/hxfeng/PAM-FlowNav/`

## HPC Environment

### Login

```bash
ssh hpc   # hpc-login-01, port 2223
ssh hpc-02  # hpc-login-02, port 2222
```

### Activate Env

```bash
source ~/anaconda3/etc/profile.d/conda.sh && conda activate pamflownav
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
unset ALL_PROXY
```

### Load CUDA

```bash
. /usr/share/modules/init/bash
module use --append /home/share/modules/modulefiles
module load cuda/12.4.1
```

### Proxy (for pip/git on HPC)

```bash
export HTTPS_PROXY=http://127.0.0.1:18990
export HTTP_PROXY=http://127.0.0.1:18990
export ALL_PROXY=""
```

### SLURM Commands

```bash
squeue --me                    # my jobs
scontrol show job <JOB_ID>     # job details + estimated start
scancel <JOB_ID>               # cancel job
scontrol ping                  # check if controllers are UP
scir-watch -s                  # GPU availability by type
scir-account -d                # compute balance (real-time)
sinfo -N -o "%5N %5t %13C %8e %7m %G"  # node details
```

### GPU Cost Reference

| GPU Type | Cost/GPU/unit | Notes |
|----------|---------------|-------|
| nvidia_a100_80gb_pcie | 16 | **Preferred** for 80GB jobs |
| a100-sxm4-80gb | 20 | Use if PCIe unavailable |
| a100-pcie-40gb | 8 | 40GB max, insufficient for JanusVLN |
| nvidia_h100 | 60 | **Avoid** unless all others occupied |

### CPU Limit Rule

Cluster enforces: total CPUs ≤ num_GPUs × 4

For 4-GPU job: `--ntasks-per-node=1 --cpus-per-task=16`

## Key Paths

```
Project root:  /home/hxfeng/PAM-FlowNav/
JanusVLN src:  /home/hxfeng/PAM-FlowNav/JanusVLN/src/
Checkpoints:   /home/hxfeng/PAM-FlowNav/data/checkpoints/misstl/
SLURM scripts: /home/hxfeng/PAM-FlowNav/scripts/slurm/
Logs:          /home/hxfeng/PAM-FlowNav/logs/
Habitat-lab:   /home/hxfeng/PAM-FlowNav/habitat-lab/ (v0.2.4, editable)
```

## Persistent Bugs & Fixes

### 1. Dead ALL_PROXY

`ALL_PROXY=http://127.0.0.1:17897` is always dead on HPC login nodes.
**Fix**: Always `unset ALL_PROXY` in SLURM scripts and before pip/git.

### 2. LD_LIBRARY_PATH

conda activate does NOT set LD_LIBRARY_PATH → TensorRT lib wins.
**Fix**: `export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH`

### 3. SLURM Absolute Paths

`${BASH_SOURCE[0]}` resolves to `/var/spool/` on compute nodes.
**Fix**: Always use absolute paths in sbatch scripts.

### 4. flash-attn Compilation

Requires CUDA_HOME set + proxy for GitHub download.
```bash
. /usr/share/modules/init/bash && module load cuda/12.4.1
HTTPS_PROXY=http://127.0.0.1:18990 TORCH_CUDA_ARCH_LIST="8.0" \
  pip install flash-attn --no-build-isolation
```

### 5. habitat-sim conda install

Use `--override-channels` + direct URLs to avoid libmamba timeout:
```bash
conda install habitat-sim==0.2.4 withbullet headless \
  -c conda-forge -c aihabitat --override-channels
```
