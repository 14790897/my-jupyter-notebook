## fid

```sh
python -m torch_fidelity.fidelity --fid --gpu 0 --input1 "C:\Users\13963\Downloads\version9\generated_images" --input2 "C:\git-program\particle_detect\auto_generate\dataset\generated_images_2"
```

35.36

### 训练数据集 fid

```sh
python -m torch_fidelity.fidelity  --fid --gpu 0 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0"
```

59.95

#### 十次运行

```sh
python -m torch_fidelity.fidelity --fid --gpu 0 --subsets 10 --subset-size 100 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0"
```

### 测试数据集 fid

```sh
python -m torch_fidelity.fidelity  --fid --gpu 0 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\test\0"
```

72.7

```sh
python -m torch_fidelity.fidelity  --fid --gpu 0 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\test\0"
```

93.04501

### 测试数据集 kid

```sh
python -m torch_fidelity.fidelity --kid --gpu 0 --kid-subset-size 100 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\test\0"
```

```sh
Kernel Inception Distance: 0.07390255581202652 ± 0.008263823044757618
kernel_inception_distance_mean: 0.07390256
kernel_inception_distance_std: 0.008263823
```

## kid

## 总共 319 张真实图片

```sh
python -m torch_fidelity.fidelity --kid --gpu 0 --kid-subset-size 100 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0"
```

### output

```sh
Kernel Inception Distance: 0.025951062282986142 ± 0.0033902050048239354
kernel_inception_distance_mean: 0.02595106
kernel_inception_distance_std: 0.003390205
```
