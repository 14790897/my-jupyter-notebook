# %% [markdown]
# # BiVAE推荐系统 - MovieLens 20M数据集（含个人评分）
# 本脚本使用Microsoft Recommenders库中的BiVAE算法（通过Cornac后端），在MovieLens 20M数据集上构建推荐系统。
# 并将用户的个人评分加入训练集，为用户生成个性化推荐。
# 
# # BiVAE简介
# BiVAE（Bilateral Variational Autoencoder）是2021年WSDM会议提出的双边变分自编码器，专为二向数据（用户-物品）设计：
# - 同时对用户侧和物品侧进行隐式表示学习
# - 对称处理用户和物品，更契合推荐场景
# - 采用变分推断，能捕捉数据不确定性
# - 支持多种似然函数（Poisson/Bernoulli/Gaussian）

# %% [code]
# 安装依赖（Kaggle环境）
%pip install -q "numpy<2.0"
%pip install -q "recommenders>=0.7.0"
%pip install -q cornac

# %% [code]
# 导入必要的库
import sys
import os
import pandas as pd
import numpy as np
import torch
import cornac

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED as DEFAULT_SEED
from recommenders.evaluation.python_evaluation import map, ndcg_at_k, precision_at_k, recall_at_k

import warnings
warnings.filterwarnings('ignore')

print(f"System: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Cornac: {cornac.__version__}")

# %% [markdown]
# # 一、加载数据并加入个人评分

# %% [code]
# 设置参数
TOP_K = 20
NUM_EPOCHS = 30
BATCH_SIZE = 128
SEED = DEFAULT_SEED

# BiVAE模型参数
LATENT_DIM = 50
ENCODER_DIMS = [100]
ACT_FUNC = "tanh"
LIKELIHOOD = "pois"  # Poisson似然，适合隐式反馈
LEARNING_RATE = 0.001

# Kaggle MovieLens 20M 数据集路径
DATASET_PATH = '/kaggle/input/datasets/organizations/grouplens/movielens-20m-dataset'

print(f"Top-K: {TOP_K}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Latent dim: {LATENT_DIM}")
print(f"Likelihood: {LIKELIHOOD}")
print(f"Dataset path: {DATASET_PATH}")

# %% [code]
# 加载MovieLens 20M评分数据（手动读取CSV）
print("正在加载MovieLens 20M评分数据...")

ratings_path = os.path.join(DATASET_PATH, 'rating.csv')
df = pd.read_csv(ratings_path)

# 重命名列以匹配recommenders库格式
df = df.rename(columns={
    'userId': 'userID',
    'movieId': 'itemID',
    'rating': 'rating',
    'timestamp': 'timestamp'
})

print(f"评分数据加载成功！形状: {df.shape}")
print("数据预览:")
display(df.head())

# %% [code]
# 加入用户的个人评分（作为新用户）
print("加入个人评分...")

# 个人评分数据
personal_ratings = [
    (115617, 5),  # Big Hero 6 (2014)
    (106696, 5),  # Frozen (2013)
    (115969, 5),  # Generation War (2013)
    (6350,   5),  # Laputa: Castle in the Sky (1986)
    (81564,  4),  # Megamind (2010)
    (1203,   4),  # 12 Angry Men (1957)
    (1291,   4),  # Spirited Away (2001)
    (152081, 4),  # Zootopia (2016)
    (920,    4),  # Gone with the Wind (1939)
    (97938,  4),  # Life of Pi (2012)
    (7099,   3),  # Nausicaä of the Valley of the Wind (1984)
    (318,    3),  # The Shawshank Redemption (1994)
]

# 新用户ID（原数据集中最大用户ID + 1）
MY_USER_ID = int(df['userID'].max()) + 1
print(f"新用户ID: {MY_USER_ID}")

# 构造新用户的评分数据
my_ratings_df = pd.DataFrame({
    'userID': [MY_USER_ID] * len(personal_ratings),
    'itemID': [itemID for itemID, _ in personal_ratings],
    'rating': [rating for _, rating in personal_ratings],
    'timestamp': [0] * len(personal_ratings)
})

print(f"个人评分记录数: {len(my_ratings_df)}")
display(my_ratings_df)

# 合并到主数据集
df = pd.concat([df, my_ratings_df], ignore_index=True)
print(f"合并后数据形状: {df.shape}")

# %% [code]
# 分割数据集（随机分割 75/25）
print("分割数据集...")
train, test = python_random_split(df, ratio=0.75, seed=SEED)

print(f"训练集大小: {len(train)}")
print(f"测试集大小: {len(test)}")
print()

# 确认新用户的评分在训练集中
my_train = train[train['userID'] == MY_USER_ID]
my_test = test[test['userID'] == MY_USER_ID]
print(f"新用户在训练集中的评分数: {len(my_train)}")
print(f"新用户在测试集中的评分数: {len(my_test)}")
print("训练集预览（新用户）:")
display(my_train.head())

# %% [markdown]
# # 二、准备Cornac数据集

# %% [code]
# 构建Cornac Dataset对象
print("构建Cornac Dataset...")

train_set = cornac.data.Dataset.from_uir(
    train.itertuples(index=False),
    seed=SEED
)

print(f"Cornac训练集:")
print(f"  - 用户数: {train_set.num_users}")
print(f"  - 物品数: {train_set.num_items}")
# print(f"  - 交互数: {train_set.num_interactions}")

# %% [markdown]
# # 三、配置和训练BiVAE模型

# %% [code]
# 创建和训练BiVAE模型
print("创建BiVAE模型...")

bivae = cornac.models.BiVAECF(
    k=LATENT_DIM,
    encoder_structure=ENCODER_DIMS,
    act_fn=ACT_FUNC,
    likelihood=LIKELIHOOD,
    n_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    seed=SEED,
    use_gpu=torch.cuda.is_available(),
    verbose=True
)

print("开始训练...")
with Timer() as train_time:
    bivae.fit(train_set)

print(f"训练完成！耗时: {train_time.interval:.2f} 秒")

# %% [markdown]
# # 四、为个人用户生成推荐

# %% [code]
# 加载电影信息
print("加载电影信息...")

movies_path = os.path.join(DATASET_PATH, 'movie.csv')
movies_df = pd.read_csv(movies_path)
movies_df = movies_df.rename(columns={
    'movieId': 'itemID',
    'title': 'title',
    'genres': 'genres'
})

print(f"电影信息加载成功！形状: {movies_df.shape}")
print("电影信息预览:")
display(movies_df.head())

# %% [code]
# 为新用户生成推荐
print(f"为用户 {MY_USER_ID} 生成 Top-{TOP_K} 推荐...")

# 使用Recommenders的predict_ranking生成推荐排序
with Timer() as pred_time:
    all_predictions = predict_ranking(
        bivae, train,
        usercol='userID', itemcol='itemID',
        remove_seen=True,
        batch_size=BATCH_SIZE
    )

print(f"推荐完成！耗时: {pred_time.interval:.2f} 秒")

# %% [code]
# 展示个人用户的推荐结果（含电影名称）
print(f"=== 为用户 {MY_USER_ID} 的 Top-{TOP_K} 推荐 ===")
print()

user_preds = all_predictions[all_predictions['userID'] == MY_USER_ID].copy()
user_preds = user_preds.sort_values('prediction', ascending=False).head(TOP_K)

user_recs = user_preds.merge(movies_df[['itemID', 'title', 'genres']], on='itemID', how='left')
display(user_recs[['itemID', 'title', 'genres', 'prediction']])
print()

# 展示用户的训练集评分（历史偏好）
print(f"=== 用户 {MY_USER_ID} 的历史评分（训练集）===")
user_history = my_train.merge(movies_df[['itemID', 'title', 'genres']], on='itemID', how='left')
display(user_history[['itemID', 'title', 'genres', 'rating']])
print()

# 展示用户的测试集评分（验证推荐是否合理）
if len(my_test) > 0:
    print(f"=== 用户 {MY_USER_ID} 的测试集评分（验证集）===")
    user_test = my_test.merge(movies_df[['itemID', 'title', 'genres']], on='itemID', how='left')
    display(user_test[['itemID', 'title', 'genres', 'rating']])

# %% [markdown]
# # 五、评估模型（采样测试集，避免内存溢出）

# %% [code]
# 采样测试集用户进行评估（避免全量计算爆内存）
print("采样测试集用户进行评估（避免内存溢出）...")

SAMPLE_USERS = 1000

# 随机选取测试集中的部分用户
test_user_sample = test['userID'].drop_duplicates().sample(n=SAMPLE_USERS, random_state=SEED)
test_sample = test[test['userID'].isin(test_user_sample)].copy()

print(f"测试集总用户数: {test['userID'].nunique()}")
print(f"采样用户数: {SAMPLE_USERS}")
print(f"采样数据量: {len(test_sample)}")

# 为采样用户生成推荐（使用predict_ranking）
print("为采样用户生成推荐...")
with Timer() as sample_pred_time:
    sample_predictions = predict_ranking(
        bivae, train,
        usercol='userID', itemcol='itemID',
        remove_seen=True,
        batch_size=BATCH_SIZE
    )
    # 只保留采样用户
    sample_predictions = sample_predictions[sample_predictions['userID'].isin(test_user_sample)]

print(f"推荐完成！耗时: {sample_pred_time.interval:.2f} 秒")

# %% [code]
# 评估模型性能（基于采样数据）
print("评估模型性能（基于采样数据）...")
print()

eval_map = map(test_sample, sample_predictions, k=TOP_K)
eval_ndcg = ndcg_at_k(test_sample, sample_predictions, k=TOP_K)
eval_precision = precision_at_k(test_sample, sample_predictions, k=TOP_K)
eval_recall = recall_at_k(test_sample, sample_predictions, k=TOP_K)

print("=== 评估结果（采样测试集，1000用户）===")
print(f"MAP@{TOP_K}:\t{eval_map:.6f}")
print(f"NDCG@{TOP_K}:\t{eval_ndcg:.6f}")
print(f"Precision@{TOP_K}:\t{eval_precision:.6f}")
print(f"Recall@{TOP_K}:\t{eval_recall:.6f}")

# %% [markdown]
# # 六、总结

# %% [code]
# 打印完整总结
print("="*60)
print("BiVAE推荐系统 - 实验总结")
print("="*60)
print()
print("【数据集】")
print(f"  - 类型: MovieLens 20M")
print(f"  - 用户数: {train_set.num_users}")
print(f"  - 物品数: {train_set.num_items}")
print(f"  - 训练交互数: {len(train)}")
print(f"  - 测试交互数: {len(test)}")
print()
print("【个人用户】")
print(f"  - 用户ID: {MY_USER_ID}")
print(f"  - 训练集评分数: {len(my_train)}")
print(f"  - 测试集评分数: {len(my_test)}")
print()
print("【模型配置】")
print(f"  - 算法: BiVAE (Bilateral Variational Autoencoder)")
print(f"  - 隐式维度: {LATENT_DIM}")
print(f"  - 编码器维度: {ENCODER_DIMS}")
print(f"  - 激活函数: {ACT_FUNC}")
print(f"  - 似然函数: {LIKELIHOOD}")
print(f"  - 训练轮数: {NUM_EPOCHS}")
print(f"  - 学习率: {LEARNING_RATE}")
print(f"  - GPU加速: {torch.cuda.is_available()}")
print()
print("【评估结果（采样测试集，1000用户）】")
print(f"  - MAP@{TOP_K}: {eval_map:.4f}")
print(f"  - NDCG@{TOP_K}: {eval_ndcg:.4f}")
print(f"  - Precision@{TOP_K}: {eval_precision:.4f}")
print(f"  - Recall@{TOP_K}: {eval_recall:.4f}")
print()
print("【个人推荐】")
print(f"  - 已为用户 {MY_USER_ID} 生成 Top-{TOP_K} 推荐")
print(f"  - 推荐结果见上方「推荐结果展示」部分")
print("="*60)
