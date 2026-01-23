## FID/KID 计算方法说明

本文档包含两种计算 FID 的方法：

1. **方法1**: 使用 `torch_fidelity` 命令行 - 计算两个不同数据集之间的FID
2. **方法2**: 使用 `calculate_baseline_fid.py` 脚本（推荐） - 计算单个数据集内部的基线FID

### 基线FID的意义
- 基线FID = 将数据集随机分成两半，计算它们之间的FID
- 代表该数据集的"最佳可能FID"
- 生成器的FID应该接近或低于基线FID

---

## fid

### 生成图像 vs 真实图像 fid

#### 方法1: 使用 torch_fidelity 命令行
```sh
python -m torch_fidelity.fidelity --fid --gpu 0 --input1 "C:\Users\13963\Downloads\version9\generated_images" --input2 "C:\git-program\particle_detect\auto_generate\dataset\generated_images_2"
```

35.36

#### 说明
这个命令计算两组生成图像之间的FID，可以用来比较不同版本的生成器性能。

### 训练数据集 fid (内部基线)

#### 方法1: 使用 torch_fidelity 命令行
```sh
python -m torch_fidelity.fidelity  --fid --gpu 0 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0"
```

59.95

#### 方法2: 使用 calculate_baseline_fid.py 脚本（推荐）
```sh
# 计算单个数据集的内部基线FID（将数据集分成两半计算）
python calculate_baseline_fid.py --data_path "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0"

# 或者计算另一个数据集
python calculate_baseline_fid.py --data_path "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0"

# 计算合并后的训练集基线FID（需要先将两个数据集合并到一个目录）
python calculate_baseline_fid.py --data_path "./train/data"
```

#### 十次运行

python .\fid\fid.py

65.96

### 测试数据集 fid

#### 方法1: 使用 torch_fidelity 命令行（训练集 vs 测试集）
```sh
python -m torch_fidelity.fidelity  --fid --gpu 0 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\test\0"
```

72.7

```sh
python -m torch_fidelity.fidelity  --fid --gpu 0 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\test\0"
```

93.04501

#### 方法2: 使用 calculate_baseline_fid.py 脚本（测试集内部基线）
```sh
# 计算测试集内部的基线FID（将测试集分成两半计算）
python calculate_baseline_fid.py --data_path "C:\git-program\particle_detect\auto_generate\dataset\test\0"

# 这将得到测试集本身的最佳可能FID值
# 生成器在测试集上的FID应该与此值对比
```
### kid不使用
<!-- ### 测试数据集 kid

```sh
python -m torch_fidelity.fidelity --kid --gpu 0 --kid-subset-size 100 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\test\0"
```

```sh
Kernel Inception Distance: 0.07390255581202652 ± 0.008263823044757618
kernel_inception_distance_mean: 0.07390256
kernel_inception_distance_std: 0.008263823
``` -->

<!-- ## 总共 319 张真实图片

```sh
python -m torch_fidelity.fidelity --kid --gpu 0 --kid-subset-size 100 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0" -->
<!-- ```

### output

```sh
Kernel Inception Distance: 0.025951062282986142 ± 0.0033902050048239354
kernel_inception_distance_mean: 0.02595106
kernel_inception_distance_std: 0.003390205
``` -->


python -m torch_fidelity.fidelity --kid --gpu 0 --kid-subset-size 100 --input1 "C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0" --input2 "C:\git-program\particle_detect\auto_generate\dataset\generated_images_2"

<!-- Kernel Inception Distance: 0.39059902521306833 ± 0.004857279021666483                                                                                                        
kernel_inception_distance_mean: 0.390599
kernel_inception_distance_std: 0.004857279 -->



## SAGAN_now.py可以绘制attention map

![alt text](image.png)