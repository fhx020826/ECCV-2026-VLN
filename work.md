# Work Progress — ECCV-2026 DifNav

## 2026-03-17 — 环境搭建与数据准备

### 完成
- [x] 创建 ECCV-2026 Git 仓库，推送至 `fhx020826/ECCV-2026-VLN`
- [x] 克隆 ETPNav (MarSaKi/ETPNav) 至 `ETPNav/`
- [x] 下载 R2R_VLNCE_v1-3_preprocessed (250MB, 官方 VLN-CE 数据)
- [x] 生成 BERTidx 格式 (4 splits: train 10819, val_seen 778, val_unseen 1839, test 3408)
- [x] 更新 config 路径至 v1-3 数据集 (r2r_vlnce.yaml × 2)
- [x] 下载 wp_pred checkpoint (193MB, 路径点预测器)
- [x] 安装 habitat-sim 0.1.7 headless 至 vlnce conda env (Python 3.6)
- [x] 创建 coding.md / claude.md / user.md / work.md

### 进行中
- [ ] DUET 预训练数据 (Dropbox, ~9-10GB): 下载中，curl PID=2791033
- [ ] LXMERT HuggingFace 版本 (`lxmert_hf.bin`): wget PID=3864629
- [ ] torch 1.9.1 + cudatoolkit=11.1 安装至 vlnce env: conda bg task

### 待办
- [ ] 验证 LXMERT HF 权重 key 名称，编写 `convert_lxmert_hf.py` 转换脚本
- [ ] 下载 CLIP 预计算特征 (`CLIP-ViT-B-32-views-habitat.hdf5`) — Google Drive
- [ ] 下载 depth 预计算特征 (`ddppo_resnet50_depth_features.hdf5`) — Google Drive
- [ ] 获取 `download_mp.py` 并提交 MP3D 场景数据下载作业 (CPU sbatch)
- [ ] 完善 vlnce 环境: habitat-lab v0.1.7 + transformers 4.12.5 + CLIP + 其他
- [ ] 编写 ETPNav 预训练 SLURM 脚本 (2×A100-80GB, ~2-3天)
- [ ] 运行 ETPNav 预训练 → model_step_82500.pt
- [ ] 运行 ETPNav VLN-CE finetune → baseline 结果
- [ ] 开始 DifNav 扩散策略模块开发

## 关键问题记录
- **ETPNav etp_ckpt.zip 已被作者意外删除** (2026-01-19)，无法直接获取预训练权重，必须自己跑完整预训练
- **UNC NLP LXMERT 服务器** 返回不完整文件，已改为从 HuggingFace 下载
- **HPC 无 CUDA 11.1 系统版本**，使用 conda-managed cudatoolkit=11.1 解决
- MP3D 场景数据需要用户提供 `download_mp.py`（Matterport 授权后从邮件获取）
