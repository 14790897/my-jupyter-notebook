import os, re, random, shutil, subprocess
import numpy as np
from pathlib import Path

input1 = r"C:\git-program\particle_detect\auto_generate\dataset\efficient_net_data_me\cropped_objects\0"
input2 = (
    r"C:\git-program\particle_detect\auto_generate\dataset\efficient2\cropped_objects\0"
)
temp1, temp2 = Path("temp_sample1"), Path("temp_sample2")
temp1.mkdir(exist_ok=True)
temp2.mkdir(exist_ok=True)

N = 100  # 每次抽样张数
R = 10  # 次数
seed = 2025  # 可改，便于复现
random.seed(seed)


def pick(src, dst, n):
    imgs = [
        f
        for f in os.listdir(src)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
    ]
    if len(imgs) < n:
        raise RuntimeError(f"{src} 少于 {n} 张")
    for f in dst.glob("*"):
        f.unlink()
    for name in random.sample(imgs, n):
        shutil.copy(os.path.join(src, name), dst / name)


pat = re.compile(r"frechet_inception_distance:\s*([0-9.]+)", re.I)
scores = []

for i in range(R):
    pick(input1, temp1, N)
    pick(input2, temp2, N)
    cmd = [
        "python",
        "-m",
        "torch_fidelity.fidelity",
        "--fid",
        "--gpu",
        "0",
        "--input1",
        str(temp1),
        "--input2",
        str(temp2),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = pat.search(out)
    if not m:
        raise RuntimeError(f"未解析到FID，原始输出：\n{out}")
    fid = float(m.group(1))
    scores.append(fid)
    print(f"[{i+1}/{R}] FID = {fid:.5f}")

print("\n结果：", [round(s, 5) for s in scores])
print(f"平均 = {np.mean(scores):.5f}，标准差 = {np.std(scores, ddof=1):.5f}")


# 删除临时文件夹
shutil.rmtree(temp1)
shutil.rmtree(temp2)