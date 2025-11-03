#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 calculate_baseline_fid.py 脚本

这个脚本演示如何使用 calculate_baseline_fid.py
"""

import os
import subprocess
import sys


def test_calculate_baseline_fid():
    """测试基线 FID 计算脚本"""
    
    print("=" * 60)
    print("测试 Baseline FID 计算脚本")
    print("=" * 60)
    
    # 示例 1: 基本使用
    print("\n示例 1: 基本使用")
    print("-" * 60)
    data_path = "./real_images_64x64_for_fid"
    
    if os.path.exists(data_path):
        cmd = ["python", "calculate_baseline_fid.py", "--data_path", data_path]
        print(f"命令: {' '.join(cmd)}")
        subprocess.run(cmd)
    else:
        print(f"⚠️  数据路径不存在: {data_path}")
        print("   请修改路径或创建测试数据集")
    
    # 示例 2: 静默模式
    print("\n示例 2: 静默模式（只输出 FID 值）")
    print("-" * 60)
    
    if os.path.exists(data_path):
        cmd = ["python", "calculate_baseline_fid.py", 
               "--data_path", data_path, "--quiet"]
        print(f"命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(f"FID 值: {result.stdout.strip()}")
    else:
        print(f"⚠️  数据路径不存在: {data_path}")
    
    # 示例 3: 保存到文件
    print("\n示例 3: 保存结果到文件")
    print("-" * 60)
    
    if os.path.exists(data_path):
        output_file = "baseline_fid_result.txt"
        cmd = ["python", "calculate_baseline_fid.py", 
               "--data_path", data_path, 
               "--output", output_file]
        print(f"命令: {' '.join(cmd)}")
        subprocess.run(cmd)
        
        if os.path.exists(output_file):
            print(f"\n✅ 结果已保存到: {output_file}")
            print("\n文件内容:")
            with open(output_file, 'r') as f:
                print(f.read())
    else:
        print(f"⚠️  数据路径不存在: {data_path}")
    
    # 示例 4: 在 Python 中导入使用
    print("\n示例 4: 在 Python 代码中使用")
    print("-" * 60)
    
    if os.path.exists(data_path):
        try:
            from calculate_baseline_fid import calculate_baseline_fid
            
            print("正在计算基线 FID...")
            baseline_fid = calculate_baseline_fid(
                data_path=data_path,
                test_split=0.5,
                seed=42,
                verbose=False
            )
            
            if baseline_fid is not None:
                print(f"\n✅ 基线 FID: {baseline_fid:.4f}")
                print(f"\n生成器性能目标:")
                print(f"  - 优秀: < {baseline_fid:.2f}")
                print(f"  - 良好: < {baseline_fid * 1.5:.2f}")
                print(f"  - 一般: < {baseline_fid * 2:.2f}")
            else:
                print("❌ 计算失败")
                
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            print("   请确保 calculate_baseline_fid.py 在当前目录")
    else:
        print(f"⚠️  数据路径不存在: {data_path}")
    
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)


if __name__ == "__main__":
    test_calculate_baseline_fid()
