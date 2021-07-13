English | [简体中文](./README.cn.md)

# Description

Play Contra based on PPO algorithm. This project is modified from a Super Mario Bros deep reinforcement learning project. The project can be used for testing if you are interested. Good ideas are welcome.

# Environment

## 1. System library
You need to install freeglut or NVIDIA CUDA(optional), as follows:
- **Ubuntu**
```bash
apt install -y freeglut3 freeglut3-dev
```
- **CentOS**
```bash
yum install -y freeglut freeglut-devel
```

## 2. Python and Modules
You need to install python 3.7 or 3.8, and some necessary modules, as follows:
```bash
pip install gym-retro torch numpy opencv-python pyglet==1.5.0 tensorboard
```

## 3. Game ROM
Contra rom files and sha files, you need to use openAI's gym-retro-integration.
```bash
cp -Rp retro/data/experimental retro/data/stable
```

###### :point_right: **Tips: The gym-retro-Integration tool is available under linux through the gym-retro source compilation. ROM file maybe you can find from https://romhustler.org.**

# Toolkit

## 1. Train
**For example:**
```bash
python train.py --game Contra-Nes --state Level1 --processes 6
```

## 2. Test
**For example:**
```bash
python test.py --game Contra-Nes --state Level1 --from_model *.(pass|save)
```

## 3. Evaluate
**For example:**
```bash
python evaluate.py --game Contra-Nes --state Level1 --from_dir Directory (like trained_models/2021-...)
```

###### :point_right: **Tips: The --from_model or --from_dir default reference location can be changed by --loading_path.**

# Reference

:book: https://retro.readthedocs.io/en/latest.
