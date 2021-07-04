#!/bin/bash

workspace="/root"
process=$(ps -ef|grep train.py|grep -v grep)

if [ "$process" != "" ]; then
echo "The train.py process is already running!"
exit 1
fi

export DISPLAY=:1
export PATH=$PATH:/usr/local/bin

mkdir -p $workspace/Agent-contra/log

> $workspace/Agent-contra/log/evaluate.log

cd $workspace/Agent-contra

nohup python train.py --game Contra-Nes --state Level1 --processes 6 &

if [ -e "log/evaluate.log" ]; then
gnome-terminal --geometry=100x10+0+400 -- bash -c "cd $workspace/Agent-contra; tail -f log/evaluate.log"
gnome-terminal --geometry=58x8+500+45 -- bash -c "cd $workspace/Agent-contra; watch -n 1 -d python score.py log/evaluate.log"
fi
