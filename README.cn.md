[English](./README.md) | 简体中文 

# 项目说明

基于PPO算法的魂斗罗游戏智能体。这个项目是从一个超级马里奥兄弟深度强化学习项目修改而来的。如果你对此项目感兴趣可以尝试测试，欢迎推荐好的建议和想法。

# 测试环境

## 1. 系统依赖库
如果你想可视化需要安装freeglut库用于渲染游戏画面，还有英伟达的CUDA(可选)，如下：
- **Ubuntu**
```bash
apt install -y freeglut3 freeglut3-dev
```
- **CentOS**
```bash
yum install -y freeglut freeglut-devel
```

## 2. Python环境
你需要安装python3.7或者3.8和一些必要的模块，如下:
```bash
pip install gym-retro torch numpy opencv-python pyglet==1.5.0 tensorboard
```

## 3. 游戏ROM
魂斗罗的rom文件sha文件，你需要使用openAI的gym-retro-integration工具。
```bash
cp -Rp retro/data/experimental retro/data/stable
```

###### :point_right: **提示: gym-retro-integration工具linux下你可以通过gym-retro源码编译获得。ROM文件也许你可以在 https://romhustler.org 找到。**

# 工具说明

## 1. 训练工具
**例如:**
```bash
python train.py --game Contra-Nes --state Level1 --processes 6
```

## 2. 测试工具
**例如:**
```bash
python test.py --game Contra-Nes --state Level1 --from_model *.(pass|save)
```

## 3. 评估工具
**例如:**
```bash
python evaluate.py --game Contra-Nes --state Level1 --from_dir 文件夹（训练产生的trained_models/2021-...）
```

###### :point_right: **提示: 参数--from_model或者--from_dir的默认引用位置可以使用--loading_path参数临时指定。**

# 参考说明

:book: https://retro.readthedocs.io/en/latest.
