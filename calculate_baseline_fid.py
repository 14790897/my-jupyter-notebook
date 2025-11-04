#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算数据集内部的基线FID（Baseline FID）

这个脚本将数据集分成两半，计算它们之间的FID分数，
作为评估GAN生成质量的参考基线。

使用方法:
    python calculate_baseline_fid.py --data_path ./real_images_64x64_for_fid
    python calculate_baseline_fid.py --data_path ./train/data --split 0.5 --seed 42
"""

import argparse
import glob
import os
import random
import shutil
import sys
import warnings
from pathlib import Path

# 过滤 torch_fidelity 中的 TypedStorage 弃用警告
warnings.filterwarnings('ignore', category=UserWarning, message='.*TypedStorage is deprecated.*')

from torch_fidelity import calculate_metrics


def calculate_baseline_fid(data_path, test_split=0.5, seed=42, verbose=True):
    """
    计算数据集内部的FID和KID（将数据集分成两半并计算它们之间的FID和KID）
    这可以作为FID/KID评估的基线参考值
    
    参数:
        data_path (str): 数据集路径
        test_split (float): 用于第二个子集的比例（默认0.5，即对半分）
        seed (int): 随机种子，用于确保可重复性
        verbose (bool): 是否打印详细信息
    
    返回:
        tuple: (baseline_fid, baseline_kid)，数据集内部的FID和KID分数，如果计算失败返回None
    """
    if verbose:
        print("\n" + "=" * 60)
        print("📊 Calculating Baseline FID and KID (Dataset Internal)")
        print("=" * 60)
    
    # 检查路径是否存在
    if not os.path.exists(data_path):
        print(f"❌ Error: Data path does not exist: {data_path}")
        return None
    
    # 获取所有图像文件
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.gif']
    all_images = []
    
    for ext in image_extensions:
        pattern = os.path.join(data_path, ext)
        all_images.extend(glob.glob(pattern))
        # 同时检查大写扩展名
        pattern_upper = os.path.join(data_path, ext.upper())
        all_images.extend(glob.glob(pattern_upper))
    
    # 去重
    all_images = list(set(all_images))
    total_images = len(all_images)
    
    if verbose:
        print(f"📁 Data path: {data_path}")
        print(f"📊 Total images found: {total_images}")
    
    # 检查数据集大小
    if total_images < 100:
        print("⚠️  Warning: Dataset too small for reliable FID calculation")
        print("   Recommended: at least 100 images")
        if total_images < 50:
            print("❌ Error: Too few images (minimum 50 required)")
            return None
    
    # 检查分割比例
    if not (0.1 <= test_split <= 0.9):
        print(f"❌ Error: Invalid split ratio: {test_split}")
        print("   Split ratio should be between 0.1 and 0.9")
        return None
    
    # 随机打乱并分割数据集
    random.seed(seed)
    random.shuffle(all_images)
    
    split_idx = int(total_images * test_split)
    subset1_images = all_images[:split_idx]
    subset2_images = all_images[split_idx:]
    
    if verbose:
        print(f"🔀 Random seed: {seed}")
        print(f"📦 Subset 1: {len(subset1_images)} images ({(1-test_split)*100:.1f}%)")
        print(f"📦 Subset 2: {len(subset2_images)} images ({test_split*100:.1f}%)")
    
    # 检查子集大小
    if len(subset1_images) < 50 or len(subset2_images) < 50:
        print("⚠️  Warning: One or both subsets have fewer than 50 images")
        print("   This may lead to unreliable FID estimates")
    
    # 创建临时目录
    temp_dir = Path(data_path).parent / 'temp_fid_baseline'
    subset1_dir = temp_dir / 'subset1'
    subset2_dir = temp_dir / 'subset2'
    
    # 清理旧的临时目录
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    
    subset1_dir.mkdir(parents=True, exist_ok=True)
    subset2_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("📂 Creating temporary subsets...")
    
    try:
        # 复制文件到临时目录
        for img_path in subset1_images:
            dest = subset1_dir / Path(img_path).name
            shutil.copy2(img_path, dest)
        
        for img_path in subset2_images:
            dest = subset2_dir / Path(img_path).name
            shutil.copy2(img_path, dest)
        
        if verbose:
            print("🔢 Calculating FID and KID between two subsets...")
            print("   (This may take a few minutes...)")
        
        # 动态设置 KID 子集大小（不能超过最小子集的样本数）
        min_subset_size = min(len(subset1_images), len(subset2_images))
        kid_subset_size = min(1000, min_subset_size)  # 使用较小的值
        
        if verbose and kid_subset_size < 1000:
            print(f"   ⚠️  Small dataset detected: using KID subset size = {kid_subset_size}")
        
        # 计算 FID 和 KID
        metrics = calculate_metrics(
            input1=str(subset1_dir),
            input2=str(subset2_dir),
            cuda=True,  # 如果有GPU则使用
            fid=True,
            kid=True,
            kid_subset_size=kid_subset_size,  # 根据数据集大小动态调整
            verbose=False
        )
        
        baseline_fid = metrics['frechet_inception_distance']
        baseline_kid = metrics['kernel_inception_distance_mean']
        baseline_kid_std = metrics['kernel_inception_distance_std']
        
        if verbose:
            print("\n" + "=" * 60)
            print(f"✅ Baseline FID (Dataset Internal): {baseline_fid:.4f}")
            print(f"✅ Baseline KID (Dataset Internal): {baseline_kid:.4f} ± {baseline_kid_std:.4f}")
            print("=" * 60)
            print("\n📊 Interpretation:")
            print("   • This represents the 'best possible' FID/KID for this dataset")
            print("   • Your generator should aim to achieve metrics close to or below these values")
            
            # 提供解释
            if baseline_fid < 20:
                print("   • 🟢 Low baseline FID: Dataset is very consistent")
            elif baseline_fid < 50:
                print("   • 🟡 Medium baseline FID: Dataset has moderate diversity")
            else:
                print("   • 🔴 High baseline FID: Dataset has high diversity or quality variance")
            
            if baseline_kid < 0.01:
                print("   • 🟢 Low baseline KID: Dataset is very consistent")
            elif baseline_kid < 0.05:
                print("   • 🟡 Medium baseline KID: Dataset has moderate diversity")
            else:
                print("   • 🔴 High baseline KID: Dataset has high diversity or quality variance")
            
            print("\n💡 Guidelines:")
            print(f"   • Generator FID < {baseline_fid:.2f}: 🎉 Excellent!")
            print(f"   • Generator FID < {baseline_fid * 1.5:.2f}: ✓ Good")
            print(f"   • Generator FID < {baseline_fid * 2:.2f}: ⚠️  Needs improvement")
            print(f"   • Generator KID < {baseline_kid:.4f}: 🎉 Excellent!")
            print(f"   • Generator KID < {baseline_kid * 1.5:.4f}: ✓ Good")
            print(f"   • Generator KID < {baseline_kid * 2:.4f}: ⚠️  Needs improvement")
            print("=" * 60 + "\n")
        
        return baseline_fid, baseline_kid
        
    except Exception as e:
        print(f"❌ Error calculating baseline FID: {e}")
        import traceback
        if verbose:
            traceback.print_exc()
        return None
        
    finally:
        # 清理临时目录
        if temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
                if verbose:
                    print("🧹 Cleaned up temporary files")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up temporary directory: {e}")


def main():
    """命令行接口"""
    parser = argparse.ArgumentParser(
        description='Calculate baseline FID for a dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python calculate_baseline_fid.py --data_path ./real_images_64x64_for_fid
  
  # Custom split ratio and seed
  python calculate_baseline_fid.py --data_path ./train/data --split 0.6 --seed 123
  
  # Quiet mode (only output FID value)
  python calculate_baseline_fid.py --data_path ./images --quiet
        """
    )
    
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to the dataset directory containing images'
    )
    
    parser.add_argument(
        '--split',
        type=float,
        default=0.5,
        help='Split ratio for second subset (default: 0.5, i.e., 50-50 split)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet mode: only output the FID value'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file to save the result (optional)'
    )
    
    args = parser.parse_args()
    
    # 计算基线 FID 和 KID
    result = calculate_baseline_fid(
        data_path=args.data_path,
        test_split=args.split,
        seed=args.seed,
        verbose=not args.quiet
    )
    
    # 处理结果
    if result is None:
        sys.exit(1)
    
    baseline_fid, baseline_kid = result
    
    # 如果是quiet模式，只输出数值
    if args.quiet:
        print(f"FID: {baseline_fid:.4f}, KID: {baseline_kid:.4f}")
    
    # 保存到文件
    if args.output:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write("Baseline FID and KID Calculation Results\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Dataset Path: {args.data_path}\n")
                f.write(f"Split Ratio: {args.split}\n")
                f.write(f"Random Seed: {args.seed}\n")
                f.write(f"\nBaseline FID: {baseline_fid:.4f}\n")
                f.write(f"Baseline KID: {baseline_kid:.4f}\n\n")
                f.write("=" * 60 + "\n")
            
            if not args.quiet:
                print(f"💾 Results saved to: {args.output}")
                
        except Exception as e:
            print(f"⚠️  Warning: Could not save results to file: {e}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
