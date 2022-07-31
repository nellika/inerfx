# RGB-only six degrees of freedom pose estimation with neural radiance fields

Special thanks to:
- [https://github.com/bmild/nerf](https://github.com/bmild/nerf)
- [https://github.com/yenchenlin/nerf-pytorch/](https://github.com/yenchenlin/nerf-pytorch)
- [https://github.com/salykovaa/inerf](https://github.com/salykovaa/inerf)

## Example run
```
python main.py --config configs/lego.txt --model_name dotted_gripper_200000 --obj_name dotted_gripper --kernel_size 5 --batch_size 512 --obs_img_num 0 --delta_phi 7 --delta_theta 2 --delta_psi 5 --experiment test_experiment

```

## Used objects & architecture
![Used object](/assets/imgs/tilted_merged_5.png)
![Used architecture](/assets/imgs/full_work_arch_whitebg.png)

## Connected links
- [Example runs](https://bit.ly/3yA0N2J)
- [Data generation](https://github.com/nellika/synth-data-generator)
- [Parse logs](https://github.com/nellika/parse-thesis-results)

