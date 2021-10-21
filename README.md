English | [简体中文](./README.cn.md)

# Description

Play Contra based on PPO with RND algorithm. This project is a testing deep reinforcement learning project. The project can be used for testing if you are interested. Good ideas are welcome.

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
You need to use openAI's gym-retro-integration gets rom files and sha files and merge with the Contra-Nes catalog to form a complete environment.
```bash
cp -Rp retro/data/experimental/Contra-Nes "python's library directory"/site-packages/retro/data/stable
```

###### :point_right: **Tips: The gym-retro-Integration tool is available under linux through the gym-retro source compilation. ROM file maybe you can find from https://romhustler.org.**

# Toolkit

## 1. Train
**For example:**
```bash
<<<<<<< HEAD
python train.py --game Contra-Nes --state Level1 --processes 6 (optional --render)
=======
python train.py --game Contra-Nes --state Level1 --processes 6
>>>>>>> origin/master
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

# Reference

:book: https://retro.readthedocs.io/en/latest.
