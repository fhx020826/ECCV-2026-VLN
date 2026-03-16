# User Preferences & Key Commands

## Environment
- Proxy port: 17897 (local SSH forward)
  - `export HTTPS_PROXY=http://127.0.0.1:17897`
  - curl: `curl --proxy http://127.0.0.1:17897`
  - wget: `HTTPS_PROXY=http://127.0.0.1:17897 wget ...`
  - gdown: `HTTPS_PROXY=http://127.0.0.1:17897 python -m gdown ...`
- Conda base: `~/anaconda3`
- Key envs:
  - `vlnce`: Python 3.6, habitat-sim 0.1.7 — ETPNav training
  - `eccv`: Python 3.8, torch 2.x — preprocessing / new modules

## HPC Commands
```bash
scir-watch -s                          # GPU availability overview
sinfo -N -o "%5N  %5t  %13C  %8O  %8e  %7m  %G"  # node detail
squeue --me                            # my jobs
scancel <job_id>                       # cancel job
scir-account -d                        # check compute points (realtime)
scir-watch <node> gpustat              # GPU usage on specific node
```

## Preferred GPU (cost order)
1. `tesla_v100-sxm2-16gb` cost=4 (16GB, good for small jobs)
2. `tesla_v100-pcie-32gb` cost=7 (32GB)
3. `tesla_v100s-pcie-32gb` cost=8 (32GB)
4. `nvidia_a100_80gb_pcie` cost=16 (80GB, **default for ETPNav training**)
5. `a100-sxm4-80gb` cost=20 (80GB)
6. **Never use** `nvidia_h100` cost=60 unless all others occupied

## Git / GitHub
- Remote: `https://github.com/fhx020826/ECCV-2026-VLN.git`
- Push via proxy: `HTTPS_PROXY=http://127.0.0.1:17897 git push origin main`
- Only commit code; never commit weights/datasets/logs

## Communication Style
- Chinese responses
- No emojis
- Concise, direct
- Report back after long jobs start (don't wait for completion)
- Estimate finish time when submitting long jobs

## Key Paths
- Project root: `/home/hxfeng/ECCV-2026`
- ETPNav code: `/home/hxfeng/ECCV-2026/ETPNav`
- Datasets: `/home/hxfeng/ECCV-2026/ETPNav/data/datasets`
- Scene data: `/home/hxfeng/ECCV-2026/ETPNav/data/scene_datasets/mp3d`
- Pretrain data: `/home/hxfeng/ECCV-2026/ETPNav/pretrain_src/datasets`
- Pretrained models: `/home/hxfeng/ECCV-2026/ETPNav/pretrained`
- HPC shared models: `/home/share/models`
