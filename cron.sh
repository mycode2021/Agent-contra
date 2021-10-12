#!/bin/bash

path="/root/Agent-contra"
eval $(date +'%d %H %M'|awk '{printf "D=$[10#%s+0]\nH=$[10#%s+0]\nM=$[10#%s+0]\n",$1,$2,$3}')
echo "$[($M+30)%60] $[(($M+30)/60+$H)%24] $[$D+(($M+30)/60+$H)/24] * * root $path/train.sh $1" >> /etc/crontab
rm -rf $path/log $path/records $path/tensorboard
read -p "Restart now, no/yes?(default no): " restart
case ${restart,,} in
"yes"|"y") reboot;;
*);;
esac
