#!/bin/bash

k=2
name=""
j=12
m=5
b=512

for i in {1..4}
do


name="comb_FAST_BRIEF_i${i}xybatch${b}_dotted"
python main.py --config configs/lego.txt --model_name dotted_gripper_200000 --obj_name dotted_gripper --kernel_size 5 --batch_size $b --obs_img_num 0 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 0_g$name
python main.py --config configs/lego.txt --model_name dotted_gripper_200000 --obj_name dotted_gripper --kernel_size 5 --batch_size $b --obs_img_num 1 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 1_g$name
python main.py --config configs/lego.txt --model_name dotted_gripper_200000 --obj_name dotted_gripper --kernel_size 5 --batch_size $b --obs_img_num 2 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 2_g$name
python main.py --config configs/lego.txt --model_name dotted_gripper_200000 --obj_name dotted_gripper --kernel_size 5 --batch_size $b --obs_img_num 3 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 3_g$name
python main.py --config configs/lego.txt --model_name dotted_gripper_200000 --obj_name dotted_gripper --kernel_size 5 --batch_size $b --obs_img_num 4 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 4_g$name
python main.py --config configs/lego.txt --model_name dotted_gripper_200000 --obj_name dotted_gripper --kernel_size 5 --batch_size $b --obs_img_num 5 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 5_g$name
python main.py --config configs/lego.txt --model_name dotted_gripper_200000 --obj_name dotted_gripper --kernel_size 5 --batch_size $b --obs_img_num 6 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 6_g$name

python main.py --config configs/lego.txt --model_name dotted_base_200000 --obj_name dotted_base --kernel_size 5 --batch_size $b --obs_img_num 0 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 0_b$name
python main.py --config configs/lego.txt --model_name dotted_base_200000 --obj_name dotted_base --kernel_size 5 --batch_size $b --obs_img_num 1 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 1_b$name
python main.py --config configs/lego.txt --model_name dotted_base_200000 --obj_name dotted_base --kernel_size 5 --batch_size $b --obs_img_num 2 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 2_b$name
python main.py --config configs/lego.txt --model_name dotted_base_200000 --obj_name dotted_base --kernel_size 5 --batch_size $b --obs_img_num 3 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 3_b$name
python main.py --config configs/lego.txt --model_name dotted_base_200000 --obj_name dotted_base --kernel_size 5 --batch_size $b --obs_img_num 4 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 4_b$name
python main.py --config configs/lego.txt --model_name dotted_base_200000 --obj_name dotted_base --kernel_size 5 --batch_size $b --obs_img_num 5 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 5_b$name
python main.py --config configs/lego.txt --model_name dotted_base_200000 --obj_name dotted_base --kernel_size 5 --batch_size $b --obs_img_num 6 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 6_b$name

python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 0 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 0_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 1 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 1_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 2 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 2_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 3 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 3_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 4 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 4_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 5 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 5_a$name
python main.py --config configs/lego.txt --model_name dotted_arm_200000 --obj_name dotted_arm --kernel_size 5 --batch_size $b --obs_img_num 6 --delta_phi $j --delta_theta $k --delta_psi $m --experiment 6_a$name


done
