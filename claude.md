# Claude Project Analysis — ECCV-2026 DifNav

## Research Goal
Reproduce ETPNav (TPAMI 2024) as baseline in VLN-CE, then extend with Diffusion Policy for continuous waypoint prediction → DifNav, submitted to ECCV 2026.

## Environment Notes
- Default outbound proxy is now local Clash instead of SSH-forwarded `17897`
- Local Clash ports:
  - HTTP `127.0.0.1:18990`
  - SOCKS5 `127.0.0.1:18991`
  - Mixed `127.0.0.1:18993`
- Shell shortcuts loaded from `~/.bashrc`:
  - `clash`
  - `proxy`
  - `unproxy`
  - `proxy-status`

## Repository Layout
```
ECCV-2026/
├── ETPNav/                    # Baseline (cloned from MarSaKi/ETPNav)
│   ├── run.py                 # Main entry point
│   ├── run_r2r/               # R2R VLN-CE config + run scripts
│   │   ├── main.bash          # Train/eval/infer launcher
│   │   ├── iter_train.yaml    # Training hyperparams
│   │   └── r2r_vlnce.yaml     # Dataset/simulator config
│   ├── vlnce_baselines/
│   │   ├── ss_trainer_ETP.py  # Main trainer (imitation learning)
│   │   ├── models/etp/
│   │   │   ├── vilmodel_cmt.py    # GlocalTextPathNavCMT (core model)
│   │   │   └── vlnbert_init.py    # Loads pretrained checkpoint
│   │   ├── common/            # Env utils, losses, etc.
│   │   └── waypoint_pred/     # Waypoint predictor module
│   ├── habitat_extensions/    # Custom Habitat task/sensors/measures
│   │   └── config/r2r_vlnce.yaml
│   ├── pretrain_src/          # Backbone pretraining code
│   │   ├── pretrain_src/train_r2r.py   # Pretrain script
│   │   └── run_pt/r2r_pretrain_habitat.json  # Pretrain config
│   ├── bert_config/bert-base-uncased/  # BERT config + vocab only
│   └── data/
│       ├── datasets/          # R2R VLN-CE annotations
│       ├── wp_pred/           # Waypoint predictor checkpoint
│       └── scene_datasets/mp3d/  # MP3D .glb scenes (TBD)
├── scripts/
│   ├── data_download/         # Download helpers
│   └── slurm/                 # SLURM sbatch scripts
├── environments/setup_vlnce.sh
├── coding.md / claude.md / user.md / work.md
└── README.md
```

## Core Model: GlocalTextPathNavCMT
File: `vlnce_baselines/models/etp/vilmodel_cmt.py`

Architecture layers:
- `bert.lang_encoder` — 9-layer BERT transformer for instruction encoding
- `bert.pano_encoder` — 2-layer transformer for panoramic image features
- `bert.global_encoder` — 4-layer cross-modal transformer (x_layers: text↔graph)
- Action head: waypoint selection over topological map candidates

Input features:
- RGB: CLIP-ViT-B/32 (512-dim) from `precompute_img_features/`
- Depth: DDPPO ResNet50 (128-dim)
- Angle features: 4-dim heading/elevation encoding
- Graph position: relative spatial relations (sprels)

## Training Pipeline

### Stage 1: Backbone Pretraining (100K steps)
- Script: `pretrain_src/pretrain_src/train_r2r.py`
- Config: `run_pt/r2r_pretrain_habitat.json`
- Init: LXMERT weights (`pretrain_src/datasets/pretrained/LXMERT/model_LXRT.pth`)
- Data: R2R discrete annotations (DUET Dropbox) + CLIP precomputed features
- Tasks: MLM (masked language model) + SAP (single-step action prediction)
- Output: `model_step_82500.pt` (checkpoint at step 82500)
- GPU: 2× A100-80GB, ~2-3 days

### Stage 2: VLN-CE Finetuning (15K iters)
- Script: `run.py` via `run_r2r/main.bash train`
- Config: `run_r2r/iter_train.yaml`
- Init: `pretrained/ETP/mlm.sap_r2r/ckpts/model_step_82500.pt`
- Data: `R2R_VLNCE_v1-3_preprocessed_BERTidx/{split}/{split}_bertidx.json.gz`
- GPU: 2× A100-80GB, ~1 day
- Output: `data/logs/checkpoints/{exp_name}/ckpt.iter12000.pth` (best)

## Data Status (as of 2026-03-17)
| Component | Status |
|---|---|
| R2R_VLNCE_v1-3_preprocessed | ✓ Downloaded |
| R2R_VLNCE_v1-3_preprocessed_BERTidx | ✓ Generated |
| wp_pred checkpoint | ✓ Downloaded |
| DUET pretrain annotations (Dropbox) | Downloading (~9.4GB, still running) |
| LXMERT weights | Downloading from HuggingFace |
| CLIP precomputed features (Google Drive) | Pending |
| MP3D scene data (.glb) | Pending (need download_mp.py) |
| ETPNav finetuned checkpoint (etp_ckpt.zip) | DELETED by author, unavailable |

## Key Issues / Notes
- `etp_ckpt.zip` was accidentally deleted by author (2026-01-19) — no internet mirror found
- Must run full pretraining from LXMERT init (100K steps) to get `model_step_82500.pt`
- LXMERT from HuggingFace (`unc-nlp/lxmert-base-uncased`) has different key names than UNC release — requires conversion script (`convert_lxmert_hf.py`)
- vlnce env: Python 3.6, habitat-sim 0.1.7 ✓, torch 1.9.1 pending
- No CUDA 11.1 on HPC system; use conda-managed cudatoolkit=11.1

## Planned Extensions (DifNav)
- Replace ETPNav waypoint selection head with Diffusion Policy (DDPM)
- Conditioning: instruction embedding + visual features + graph state
- Output: 2D waypoint delta in agent-relative coordinates
- Training: same IL framework, diffusion denoising loss
