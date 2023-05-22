###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @Description : 
 # @LastEditTime: 2020-10-17 04:12:49
### 

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/home/LinHonghui/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/home/LinHonghui/anaconda3/etc/profile.d/conda.sh" ]; then
#         . "/home/LinHonghui/anaconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/home/LinHonghui/anaconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# <<< conda initialize <<<

# conda init bash
# conda activate freihand

cd src

# config_file="../configs/slgcn_20210608_s3t11_decay.py"
# config_file="../configs/msg3d_20210605_s3t11_decay.py"
# config_file="../configs/chalearn13_origin_ps.py"

# config_file="../configs/msg3d_20210531.py"
# config_file="../configs/proposed_20210530_l3s3t3.py"
config_file="../configs/test_chalearn13_origin_ps.py"

# load_path="/home/ChengJiale/GestureRecognition/IEEE_MM/checkpoints/20210605_slgcn_20210606_s3t11_decay/slgcn_20210606_s3t11_decay_0.05696505505026329.pth"

# load_path="/home/ChengJiale/GestureRecognition/IEEE_MM/checkpoints/20210531_msg3d_20210531/msg3d_20210531_0.48045316738471355.pth"
# load_path="/home/ChengJiale/GestureRecognition/IEEE_MM/checkpoints/20210531_proposed_20210531_l3s3t3_decay/proposed_20210531_l3s3t3_decay_0.5160363810435615.pth"

load_path="/home/ChengJiale/GestureRecognition/IEEE_MM/checkpoints/20210810_chalearn13_origin_ps/chalearn13_origin_ps_0.9409282700421941.pth"

# python ../tools/train.py    -config_file $config_file -device 0 1 2
# python ../tools/train.py    -config_file $config_file -device 0 -load_path $load_path
# python ../tools/train.py    -config_file $config_file -device 0 #-find_lr #-load_path $load_path #-find_lr #-load_path "/home/ChengJiale/PS_CVPR/checkpoints/20210304_chalearn16_20210304_proposed/chalearn16_20210304_proposed_0.46978153404560674.pth"

python ../tools/test.py -config_file $config_file -device 0 1 2 -load_path $load_path
