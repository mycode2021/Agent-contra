#!/bin/bash

eval $(date +'%H %M'|awk '{printf "H=%s\nM=%s\n",$1,$2}')
[ "$(egrep '.*train.sh.*' /etc/crontab)" != "" ] && \
sed -i "s/.*train.sh.*/$[($M+30)%60] $[(($M+30)/60+$H)%24] \* \* \* root \/root\/Agent-contra\/train\.sh/g" /etc/crontab || \
echo "$[(`expr $M + 0`+30)%60] $[((`expr $M + 0`+30)/60+`expr $H + 0`)%24] * * * root /root/Agent-contra/train.sh" >> /etc/crontab
rm -rf log records tensorboard utils/__pycache__ nohup.out score
read -p "Restart now, no/yes?(default no): " restart
case ${restart,,} in
"yes"|"y") reboot;;
*);;
esac
