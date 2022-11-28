#!/bin/bash
# wm_dir="outputs/cifar10/wm/jia_vit/00002_jia"
wm_dir="outputs/cifar10/wm/jia/00000_jia"
# wm_dir="outputs/cifar10/wm/blackmarks/00000_blackmarks"
# wm_dir="outputs/cifar10/wm/uchida/00000_uchida"
atk_conf="configs/cifar10/attack_configs_experiments/"
log_dir="log/"
arr=`ls configs/cifar10/attack_configs | grep yaml`
#arr=`ls configs/cifar10/attack_configs | grep -v 'combined_attack[1-9]*.yaml'`
#arr=`ls configs/cifar10/attack_configs | grep 'combined_attack[1-9]*.yaml'`
echo $arr
ary=(`echo $arr`)
# echo $ary
echo "the number of attack_config:"${#ary[@]}
for i in `seq 1 ${#ary[@]}`
do
  # echo ${ary[$i-1]}
  file=`echo ${ary[$i-1]} | sed 's/\.[^\.]*$//'`
  # echo $file
  echo "python3 steal.py --save --attack_config $atk_conf${ary[$i-1]} --wm_dir $wm_dir"
  `python3 steal.py --save --attack_config $atk_conf${ary[$i-1]} --wm_dir $wm_dir > $log_dir$file".out" 2> $log_dir$file"_error.out"`
done
