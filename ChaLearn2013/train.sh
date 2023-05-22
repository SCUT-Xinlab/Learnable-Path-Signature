###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @Description : 
 # @LastEditTime: 2020-11-30 22:00:52
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

# config_file="../configs/deform_pscnn_nonlocal_multi_find_lr.py"
# config_file="../configs/chalearn13_ps_withall.py"
# config_file="../configs/chalearn13_DecoupleGCN_WITHOUTNL_1.py"
# config_file="../configs/chalearn13_with_STEM_2s.py"
# config_file="../configs/chalearn13_with_STEM_2s_lr3e3.py"
# config_file="../configs/chalearn13_with_STEM_2s_dp05.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs256.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs64_dp05.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs64_dp06_lr3e3.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs64_dp05_lr3e3.py"
# config_file="../configs/chalearn13_Liao.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs64_dp02_lr3e3_bnhead_Toy.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs64_dp02_lr6e3_bnhead_Toy.py"


# config_file="../configs/chalearn13_with_STEM_2s_bs128_dp05_lr3e3_bnhead_Toy.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs64_dp05_lr3e3_bnhead.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs64_dp05_lr6e3_bnhead.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs32_dp05_lr6e3_bnhead.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs32_dp04_lr5e3_bnhead.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs128_dp05_lr3e3_bnhead.py"
# config_file="../configs/chalearn13_with_STEM_2s_bs64_dp05_lr1e2_bnhead.py"
config_file="../configs/chalearn13_origin_ps.py"

# config_file="../configs/chalearn13_learnable_path_5.py"
# config_file="../configs/chalearn13_learnable_path_11.py"
# config_file="../configs/test_chalearn13_learnable_path_11.py"

# load_path="/home/LinHonghui/Project/PS_CVPR_13_AR/checkpoints/20201130_chalearn13_with_STEM_2s_bs64_dp05_lr6e3_bnhead/chalearn13_with_STEM_2s_bs64_dp05_lr6e3_bnhead_0.9389592123769339.pth"
# load_path="/home/LinHonghui/Project/PS_CVPR_13_AR/checkpoints/20201129_chalearn13_with_STEM_2s_bs64_dp05_lr6e3_bnhead/chalearn13_with_STEM_2s_bs64_dp05_lr6e3_bnhead_0.9369901547116737.pth"
# load_path="/home/LinHonghui/Project/PS_CVPR_13_AR/checkpoints/20201117_chalearn13_with_STEM_2s_bs64_dp05_lr3e3/chalearn13_with_STEM_2s_bs64_dp05_lr3e3_0.9392405063291139.pth"
# load_path="/home/scutee/Jiale/TMM_Lin/checkpoints/20201130_chalearn13_with_STEM_2s_bs64_dp05_lr6e3_bnhead/chalearn13_with_STEM_2s_bs64_dp05_lr6e3_bnhead_0.9417721518987342.pth"

# load_path="/home/scutee/Jiale/TMM_Lin/checkpoints/20210805_chalearn13_learnable_path_11/chalearn13_learnable_path_11_0.9375527426160337.pth"

python ../tools/train.py    -config_file $config_file -device 0 1 2
# python ../tools/train.py    -config_file $config_file -device 0 -load_path $load_path
# python ../tools/train.py    -config_file $config_file -device 2 #-load_path "/home/xinlab/PS_CVPR_13_AR/checkpoints/20201114_chalearn13_ps_withall/chalearn13_ps_withall_temp.pth" #-find_lr

# python ../tools/test.py -config_file $config_file -device 0 -load_path $load_path
