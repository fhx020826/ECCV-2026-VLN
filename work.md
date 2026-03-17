# Work Progress — ECCV-2026 DifNav

## 2026-03-17 — 本机 Clash 代理配置

### 完成
- [x] 复用现有 `~/clash/clash` 二进制与用户提供的订阅配置 `RoI18DZE4t8l.yaml`
- [x] 将 `~/clash/config.yaml` 重写为独立本地端口，避免继续占用或依赖 `17897`
  - HTTP `18990`
  - SOCKS5 `18991`
  - Redir `18992`
  - Mixed `18993`
  - Controller `19090`
- [x] 将 `~/.bashrc` 中旧的 `17897` 自动代理逻辑替换为本地 Clash 别名
  - `clash`
  - `proxy`
  - `unproxy`
  - `proxy-status`
- [x] 配置校验通过：`~/clash/clash -t -d ~/clash`
- [x] 代理实测通过
  - `curl --proxy http://127.0.0.1:18990 https://api.ipify.org` 返回公网 IP
  - `git -c http.proxy=http://127.0.0.1:18990 -c https.proxy=http://127.0.0.1:18990 ls-remote https://github.com/openai/openai-python.git -h refs/heads/main` 成功

### 注意
- 当前 `clash` 进程 PID 记录在 `/tmp/clash_local.pid`
- 日志文件：`/home/hxfeng/clash/nohup.out`
- 由于当前环境限制，`redir` 的 UDP listener 有 warning，但 HTTP/SOCKS/Mixed 代理均正常工作，不影响日常 `git` / `curl` / `pip` 等走代理
- 旧的 `17897` 方案改为 fallback，不再作为默认出网方式

## 2026-03-17 — 环境搭建与数据准备

### 完成
- [x] 创建 ECCV-2026 Git 仓库，推送至 `fhx020826/ECCV-2026-VLN`
- [x] 克隆 ETPNav (MarSaKi/ETPNav) 至 `ETPNav/`
- [x] 下载 R2R_VLNCE_v1-3_preprocessed (250MB, 官方 VLN-CE 数据)
- [x] 生成 BERTidx 格式 (4 splits: train 10819, val_seen 778, val_unseen 1839, test 3408)
- [x] 更新 config 路径至 v1-3 数据集 (r2r_vlnce.yaml × 2)
- [x] 下载 wp_pred checkpoint (193MB, 路径点预测器)
- [x] 创建 coding.md / claude.md / user.md / work.md
- [x] 下载 LXMERT (HuggingFace `unc-nlp/lxmert-base-uncased`, 960MB)
- [x] 转换 LXMERT → ETPNav 格式 (`model_LXRT.pth`, `lxmert.` → `bert.` key 重映射)
- [x] 下载 CLIP+depth 图像特征 (Google Drive zip → CLIP-ViT-B-32-views-habitat.hdf5 509MB + ddppo 200MB)
- [x] 生成 scanvp_candview_relangles.json (10567 entries, 从 MP3D connectivity)
- [x] 生成 R2R pretrain JSONL (train 14039, val_seen 1021, val_unseen 2349)
- [x] 移除 prevalent_aug 依赖 (pretrain config 只用标准 R2R train)
- [x] 创建 pretrain_etp.sbatch + finetune_etp.sbatch (SLURM 脚本)
- [x] 编写 convert_lxmert_hf.py, gen_scanvp_cands.py, gen_pretrain_jsonl.py
- [x] **eccv-2026 conda 环境安装完成** (Python 3.8 + pytorch 1.9.1 + CUDA 11.1)
  - 通过 conda-forge channel (Tsinghua mirror) 安装 pytorch 1.9.1 + cudatoolkit 11.1
  - 通过 pip (Tsinghua PyPI) 安装: transformers 4.12.5, timm 0.5.4, scipy, h5py, jsonlines 等
  - 关键 bug fix: 需要 `export LD_LIBRARY_PATH=/home/hxfeng/anaconda3/envs/eccv-2026/lib:$LD_LIBRARY_PATH`
- [x] **提交 ETPNav 预训练 SLURM 作业 (job 84592)**
  - RUNNING on gpu17, NVIDIA A100 80GB PCIe × 2
  - Step 2500: MLM acc ~52%, SAP gacc ~62-64% ✓
  - checkpoint: `pretrained/ETP/mlm.sap_r2r/ckpts/model_step_2500.pt` 已保存

### 待办（优先级顺序）
- [x] **habitat-sim 0.1.7 安装完成** (Python 3.8 headless build, conda from aihabitat channel)
  - 使用 `conda install -c https://conda.anaconda.org/aihabitat -c https://conda.anaconda.org/conda-forge --override-channels habitat-sim=0.1.7 headless --no-update-deps`
  - 在 login 节点后台运行，无需 SLURM job
- [x] **habitat-lab v0.1.7 安装完成**
  - git clone 使用 `git -c http.proxy="" -c https.proxy=""` 绕过死代理
  - pip 安装: gym, moviepy, opencv-python, lmdb, ifcfg, webdataset==0.1.40, boto3, pytorch-transformers, msgpack-numpy, dtw, fastdtw, openai-clip, tensorboard
  - 关键 bug fix: `torch/utils/tensorboard/__init__.py` 改用 `from distutils.version import LooseVersion`
  - 关键 bug fix: ETPNav `vlnce_baselines/` 3处 `import tensorflow` 改为 try/except (tf not installed)
  - habitat-lab 路径加入 finetune_etp.sbatch 的 PYTHONPATH
- [x] **ETPNav finetune 全套 import 验证通过** (`vlnce_baselines: OK`, login node 无 GPU 属正常)

- [ ] 监控预训练到 step 82500 (目标 checkpoint)
  - 当前 (04:56 CST): step ~13000/100000, ~4 it/s, ~5.8h remaining
- [ ] 获取 `download_mp.py` → 提交 MP3D 场景数据下载作业
  - MP3D .glb 文件: `ETPNav/data/scene_datasets/mp3d/`
  - 这是 VLN-CE finetune 的必要前提
- [ ] **habitat-sim 0.1.7 安装** (finetuning 需要)
  - 计算节点有 CUDA 11.8 系统 CUDA
  - 当前 eccv-2026 env 没有 habitat-sim
  - 需要 SLURM job 安装 (不能用 proxy)
- [ ] **提交 VLN-CE finetune** (`sbatch scripts/slurm/finetune_etp.sbatch`)
  - 依赖: model_step_82500.pt + MP3D 场景 + 计算账户余额回正
- [ ] 开始 DifNav 扩散策略模块设计与实现

## 关键问题记录
- **ETPNav etp_ckpt.zip 已被作者意外删除** (2026-01-19)，无法直接获取预训练权重，必须自己跑完整预训练
- **UNC NLP LXMERT 服务器** 返回不完整文件，已改为从 HuggingFace 下载
- **MP3D 场景数据需要用户提供** `download_mp.py`（Matterport 授权后从邮件获取）
- **SLURM 账户余额耗尽**: `remain -3185.0984` — 无法提交新 job，已提交的 84592 仍在运行
- **habitat-sim conda 安装**: 必须用 `--override-channels` + 直接 URL (`https://conda.anaconda.org/aihabitat`)，否则 libmamba 超时
- **git proxy**: 系统 git config 设了死代理 (`http://127.0.0.1:17897`)，需用 `git -c http.proxy="" -c https.proxy="" clone`
- **webdataset**: 必须用 0.1.40 (0.2.x 改了 API，`wds.Dataset` 不存在)
- **torch tensorboard**: `torch/utils/tensorboard/__init__.py` 用 `from setuptools import distutils` 访问 `version` 失败，改为 `from distutils.version import LooseVersion`
- **conda 安装 pytorch 关键 bug**: conda activate 不设置 LD_LIBRARY_PATH，导致 TensorRT 的 libcublasLt.so.11 覆盖 conda 版本，缺少 `free_gemm_select` symbol。修复: 在 sbatch 中手动加 `export LD_LIBRARY_PATH=/home/hxfeng/anaconda3/envs/eccv-2026/lib:$LD_LIBRARY_PATH`
- **ALL_PROXY 全局设置**: 系统设置了 `ALL_PROXY=http://127.0.0.1:17897` 但代理已宕机。pip 安装时需要 `ALL_PROXY="" pip install ...`；conda 安装时需要 `env -u ALL_PROXY conda install ...`
- **train_r2r.py 的 --init_pretrained 参数**: 在 JSON config 中设置 (`init_pretrained: lxmert`)，不是命令行参数，不要在 sbatch 中传递

## 预训练进度
- **Job**: 84592 (etp_pretrain), RUNNING on gpu17
- **开始时间**: 2026-03-17 04:05 CST
- **日志**: `/home/hxfeng/ECCV-2026/logs/etp_pretrain_84592.out`
- **log.txt**: `ETPNav/pretrained/ETP/mlm.sap_r2r/logs/log.txt`
- **Step 2500 验证结果** (04:15 CST):
  - val_seen MLM acc: 52.82%, SAP gacc: 61.84%
  - val_unseen MLM acc: 51.87%, SAP gacc: 63.62%
