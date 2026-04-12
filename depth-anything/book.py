# %% [markdown]
# #  环境配置与安装 https://github.com/ByteDance-Seed/Depth-Anything-3

# %% [code]
from IPython.display import clear_output

# 克隆DA3官方仓库
!git clone https://github.com/ByteDance-Seed/Depth-Anything-3.git
%cd Depth-Anything-3

# 安装基础依赖
!pip install xformers 

# 按照文档以可编辑模式安装DA3
!pip install -e .
# !pip install git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
!pip install gsplat
clear_output()

# %% [code]
# 精确替换错误的变量名，修复官方 cli.py 中的 Bug
# !sed -i "s/reference_view_strategy=reference_view_strategy/ref_view_strategy=ref_view_strategy/g" /kaggle/working/Depth-Anything-3/src/depth_anything_3/cli.py
!find /kaggle/working/Depth-Anything-3/src -type f -name "*.py" -exec sed -i 's/reference_view_strategy/ref_view_strategy/g' {} +
# 确认替换成功（可选，用于打印修改后的那行代码查看）
!grep "ref_view_strategy" /kaggle/working/Depth-Anything-3/src/depth_anything_3/cli.py

# %% [markdown]
# # 视频推理

# %% [code]
import os

# 定义输入和输出路径
INPUT_VIDEO = "/kaggle/input/datasets/liuweiq/daxiaonailong/caixunkun.mp4"  # 替换为你实际的视频路径
OUTPUT_DIR = "/kaggle/working/da3_output"
os.environ["MODEL_DIR"] = "depth-anything/DA3-BASE"
# 创建输出目录
# os.makedirs(OUTPUT_DIR, exist_ok=True)
!rm -rf {OUTPUT_DIR}
# !da3 backend --model-dir depth-anything/da3-base

# %% [code]
# !git clone https://huggingface.co/depth-anything/DA3-BASE ./models/DA3-BASE
# DA3-Small

# %% [code] {"execution":{"iopub.status.busy":"2026-04-12T06:27:24.248664Z","iopub.execute_input":"2026-04-12T06:27:24.249179Z","iopub.status.idle":"2026-04-12T06:27:39.207474Z","shell.execute_reply.started":"2026-04-12T06:27:24.249140Z","shell.execute_reply":"2026-04-12T06:27:39.206706Z"}}
!rm -rf "workspace/gallery/scene"
!rm -rf "/kaggle/working/da3_output"
    # --use-backend \ linux注释必须在行首
!da3 video "/kaggle/input/datasets/liuweiq/daxiaonailong/caixunkun.mp4" \
    --fps 2 \
    --export-dir "/kaggle/working/da3_output"  \
    --export-format glb \
    --process-res-method lower_bound_resize \
    --process-res 120 \
    --model-dir depth-anything/DA3-SMALL