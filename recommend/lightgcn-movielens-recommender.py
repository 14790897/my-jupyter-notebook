# %% [markdown]
# # LightGCN推荐系统 - MovieLens数据集
# 本脚本使用Microsoft Recommenders库中的LightGCN算法，在MovieLens数据集上构建推荐系统。
# 
# # LightGCN简介
# LightGCN是2020年提出的图神经网络推荐算法，在NGCF基础上简化而来：
# - 移除特征转换和非线性激活
# - 只在用户-物品图上做邻域聚合
# - 更简单、更高效、性能更好
# 
# # 数据集
# 使用MovieLens 100K数据集（电影推荐）：
# - 943个用户，1682部电影
# - 100,000条评分记录（1-5分）
# - 经典的推荐系统基准数据集

# %% [code]
# 安装依赖（Kaggle环境已预装pandas, numpy等基础库）
%pip install -q "recommenders>=0.7.0"

# %% [code]
# 导入必要的库
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # 只显示错误信息

from recommenders.utils.timer import Timer
from recommenders.models.deeprec.models.graphrec.lightgcn import LightGCN
from recommenders.models.deeprec.DataModel.ImplicitCF import ImplicitCF
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_stratified_split
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.utils.constants import SEED as DEFAULT_SEED

import warnings
warnings.filterwarnings('ignore')

print(f"System: {sys.version}")
print(f"Pandas: {pd.__version__}")
print(f"TensorFlow: {tf.__version__}")

# %% [markdown]
# # 一、加载和分割数据

# %% [code]
# 设置参数
TOP_K = 20  # 修改：与YAML配置一致
MOVIELENS_DATA_SIZE = '100k'
EPOCHS = 1000  # 修改：与YAML配置一致
BATCH_SIZE = 4096
SEED = DEFAULT_SEED

print(f"Top-K: {TOP_K}")
print(f"Dataset: MovieLens {MOVIELENS_DATA_SIZE}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")

# %% [code]
# 加载MovieLens数据集
print("正在加载MovieLens数据集...")

df = movielens.load_pandas_df(size=MOVIELENS_DATA_SIZE)
print(f"数据加载成功！形状: {df.shape}")
print("数据预览:")
display(df.head())

# %% [code]
# 分割数据集
print("分割数据集...")
train, test = python_stratified_split(df, ratio=0.75, col_user='userID', col_item='itemID', seed=SEED)

print(f"训练集大小: {len(train)}")
print(f"测试集大小: {len(test)}")
print()

# 查看数据格式
print("训练集预览:")
display(train.head())

# %% [markdown]
# # 二、准备数据

# %% [code]
# 使用ImplicitCF准备数据
print("准备数据...")

data = ImplicitCF(train=train, test=test, seed=SEED)
print(f"用户数: {data.n_users}")
print(f"物品数: {data.n_items}")
print(f"训练交互数: {len(train)}")
print(f"测试交互数: {len(test)}")

# %% [markdown]
# # 三、配置和训练LightGCN模型

# %% [code]
np.mat = np.asmatrix  # 模拟 np.mat，让老代码能用

# 创建和训练LightGCN模型
print("创建LightGCN模型...")

# 手动创建hparams对象（避免依赖yaml文件）
import types
hparams = types.SimpleNamespace()

# model组参数（按照YAML配置）
hparams.model_type = 'lightgcn'
hparams.embed_size = 64
hparams.n_layers = 3

# train组参数（按照YAML配置）
hparams.batch_size = BATCH_SIZE
hparams.decay = 0.0001
hparams.epochs = EPOCHS
hparams.learning_rate = 0.001
hparams.eval_epoch = -1  # -1表示训练期间不评估
hparams.top_k = TOP_K

# info组参数（按照YAML配置）
hparams.save_model = False
hparams.save_epoch = 100
hparams.metrics = ["recall", "ndcg", "precision", "map"]
hparams.MODEL_DIR = './model_checkpoint'

model = LightGCN(hparams, data, seed=SEED)

print("开始训练...")
with Timer() as train_time:
    model.fit()

print(f"训练完成！耗时: {train_time.interval:.2f} 秒")

# %% [markdown]
# # 四、推荐和评估

# %% [code]
# 为测试集用户生成推荐
print("生成推荐...")
topk_scores = model.recommend_k_items(test, top_k=TOP_K, remove_seen=True)

print("推荐完成！")
print("推荐结果预览:")
display(topk_scores.head())

# %% [code]
# 评估模型性能
print("评估模型性能...")
print()

eval_map = map(test, topk_scores, k=TOP_K)
eval_ndcg = ndcg_at_k(test, topk_scores, k=TOP_K)
eval_precision = precision_at_k(test, topk_scores, k=TOP_K)
eval_recall = recall_at_k(test, topk_scores, k=TOP_K)

print("=== 评估结果 ===")
print(f"MAP@K:\t{eval_map:.6f}")
print(f"NDCG@K:\t{eval_ndcg:.6f}")
print(f"Precision@K:\t{eval_precision:.6f}")
print(f"Recall@K:\t{eval_recall:.6f}")

# %% [markdown]
# # 五、推荐结果展示

# %% [code]
# 加载电影信息（用于展示电影名称）
print("加载电影信息...")

movies_df = movielens.load_pandas_df(
    size='100k',
    header=['itemID', 'title', 'release_date', 'video_release_date', 'IMDb_URL',
           'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy',
           'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
           'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
)

print(f"电影信息加载成功！形状: {movies_df.shape}")

# %% [code]
# 展示某个用户的推荐结果
EXAMPLE_USER = test['userID'].unique()[0]
print(f"为用户 {EXAMPLE_USER} 生成推荐...")
print()

# 获取该用户的推荐结果
user_recs = topk_scores[topk_scores['userID'] == EXAMPLE_USER].copy()

if len(user_recs) > 0:
    # 添加电影名称
    user_recs = user_recs.merge(movies_df[['itemID', 'title']], on='itemID', how='left')
    
    print(f"=== 用户 {EXAMPLE_USER} 的Top-{TOP_K}推荐（含电影名称）===")
    display(user_recs[['itemID', 'title', 'prediction']])
    print()
    
    # 查看该用户在训练集中的历史交互
    user_history = train[train['userID'] == EXAMPLE_USER].copy()
    user_history = user_history.merge(movies_df[['itemID', 'title']], on='itemID', how='left')
    print(f"用户在训练集中的交互数: {len(user_history)}")
    print("训练集历史交互（含电影名称）:")
    display(user_history[['itemID', 'title', 'rating']].head(10))
    print()
    
    # 查看该用户在测试集中的真实交互（含电影名称）
    user_test = test[test['userID'] == EXAMPLE_USER].copy()
    user_test = user_test.merge(movies_df[['itemID', 'title']], on='itemID', how='left')
    print(f"用户在测试集中的交互数: {len(user_test)}")
    print("测试集真实交互（含电影名称）:")
    display(user_test[['itemID', 'title', 'rating']])
else:
    print("未能找到该用户的推荐结果")

# %% [code]
# 可视化：推荐分数分布
import matplotlib.pyplot as plt

if len(user_recs) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 推荐分数分布
    axes[0].bar(range(1, len(user_recs) + 1), user_recs['prediction'], alpha=0.7)
    axes[0].set_xlabel('Recommendation Rank', fontsize=12)
    axes[0].set_ylabel('Prediction Score', fontsize=12)
    axes[0].set_title(f'User {EXAMPLE_USER} Recommendation Scores', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 用户交互分布
    user_interaction_counts = train['userID'].value_counts()
    axes[1].hist(user_interaction_counts.values, bins=50, alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Number of Interactions', fontsize=12)
    axes[1].set_ylabel('Number of Users', fontsize=12)
    axes[1].set_title('User Interaction Distribution', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印统计信息
    print("=== Data Statistics ===")
    print(f"Average interactions per user: {user_interaction_counts.mean():.2f}")
    print(f"Median interactions per user: {user_interaction_counts.median():.2f}")
    print(f"Max interactions: {user_interaction_counts.max()}")
    print(f"Min interactions: {user_interaction_counts.min()}")

# %% [markdown]
# # 六、总结

# %% [code]
# 打印完整总结
print("="*60)
print("LightGCN推荐系统 - 实验总结")
print("="*60)
print()
print("【数据集】")
print(f"  - 类型: MovieLens {MOVIELENS_DATA_SIZE}")
print(f"  - 用户数: {data.n_users}")
print(f"  - 物品数: {data.n_items}")
print(f"  - 训练交互数: {len(train)}")
print(f"  - 测试交互数: {len(test)}")
print()
print("【模型配置】")
print(f"  - 算法: LightGCN")
print(f"  - 嵌入维度: 64")
print(f"  - GCN层数: 3")
print(f"  - 损失函数: BPR")
print(f"  - 训练轮数: {EPOCHS}")
print(f"  - 学习率: 0.005")
print()
print("【评估结果】")
print(f"  - MAP@{TOP_K}: {eval_map:.4f}")
print(f"  - NDCG@{TOP_K}: {eval_ndcg:.4f}")
print(f"  - Precision@{TOP_K}: {eval_precision:.4f}")
print(f"  - Recall@{TOP_K}: {eval_recall:.4f}")
print()
print("【下一步改进】")
print("  1. 尝试不同的嵌入维度 (32, 64, 128)")
print("  2. 调整GCN层数 (2, 3, 4)")
print("  3. 使用更大的MovieLens数据集 (1M, 10M)")
print("  4. 调参：学习率、正则化系数、批次大小")
print("  5. 与其他算法对比 (SAR, NCF)")
print("="*60)
