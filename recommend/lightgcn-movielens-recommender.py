# %% [markdown]
# # LightGCN推荐系统 - MovieLens 20M数据集（含个人评分）
# 本脚本使用Microsoft Recommenders库中的LightGCN算法，在MovieLens 20M数据集上构建推荐系统。
# 并将用户的个人评分加入训练集，为用户生成个性化推荐。
# 
# # LightGCN简介
# LightGCN是2020年提出的图神经网络推荐算法，在NGCF基础上简化而来：
# - 移除特征转换和非线性激活
# - 只在用户-物品图上做邻域聚合
# - 更简单、更高效、性能更好

# %% [code]
# 安装依赖（Kaggle环境）
%pip install -q "recommenders>=0.7.0"

# %% [code]
# 导入必要的库
import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

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
# # 一、加载数据并加入个人评分

# %% [code]
# 设置参数
TOP_K = 20
MOVIELENS_DATA_SIZE = '20m'
EPOCHS = 2
BATCH_SIZE = 4096
SEED = DEFAULT_SEED

# Kaggle MovieLens 20M 数据集路径（根据实际Kaggle输入路径调整）
DATASET_PATH = '/kaggle/input/datasets/organizations/grouplens/movielens-20m-dataset/'

print(f"Top-K: {TOP_K}")
print(f"Epochs: {EPOCHS}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Dataset path: {DATASET_PATH}")

# %% [code]
# 加载MovieLens 20M评分数据（手动读取CSV）
print("正在加载MovieLens 20M评分数据...")
df = movielens.load_pandas_df(size=MOVIELENS_DATA_SIZE)

# ratings_path = os.path.join(DATASET_PATH, 'rating.csv')
# df = pd.read_csv(ratings_path)

# # 重命名列以匹配recommenders库格式
# df = df.rename(columns={
#     'userId': 'userID',
#     'movieId': 'itemID',
#     'rating': 'rating',
#     'timestamp': 'timestamp'
# })

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

# 新用户ID（原数据集中最大用户ID + 10）
MY_USER_ID = df['userID'].max() + 10
print(f"新用户ID: {MY_USER_ID}")

# 构造新用户的评分数据
my_ratings_df = pd.DataFrame({
    'userID': [MY_USER_ID] * len(personal_ratings),
    'itemID': [itemID for itemID, _ in personal_ratings],
    'rating': [rating for _, rating in personal_ratings],
    'timestamp': [0] * len(personal_ratings)  # 个人评分无时间戳，用0填充
})

print(f"个人评分记录数: {len(my_ratings_df)}")
display(my_ratings_df)

# 合并到主数据集
df = pd.concat([df, my_ratings_df], ignore_index=True)
print(f"合并后数据形状: {df.shape}")

# %% [code]
# 分割数据集（分层分割，保证每个用户在训练集和测试集都有数据）
print("分割数据集...")
train, test = python_stratified_split(df, ratio=0.75, col_user='userID', col_item='itemID', seed=SEED)

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
# # 二、准备数据

# %% [code]
# 使用ImplicitCF准备数据（将评分转换为隐式反馈）
print("准备数据...")

data = ImplicitCF(train=train, test=test, seed=SEED)
print(f"用户数: {data.n_users}")
print(f"物品数: {data.n_items}")
print(f"训练交互数: {len(train)}")
print(f"测试交互数: {len(test)}")

# %% [markdown]
# # 三、配置和训练LightGCN模型

# %% [code]
np.mat = np.asmatrix  # NumPy 2.0 兼容补丁

# 手动创建hparams对象
import types
hparams = types.SimpleNamespace()

# model参数
hparams.model_type = 'lightgcn'
hparams.embed_size = 64
hparams.n_layers = 3

# train参数
hparams.batch_size = BATCH_SIZE
hparams.decay = 0.0001
hparams.epochs = EPOCHS
hparams.learning_rate = 0.001
hparams.eval_epoch = 10
hparams.top_k = TOP_K

# info参数
hparams.save_model = False
hparams.save_epoch = 100
hparams.metrics = ["recall", "ndcg", "precision", "map"]
hparams.MODEL_DIR = './model_checkpoint'

model = LightGCN(hparams, data, seed=SEED)
print("模型创建成功！")

# %% [code]
# 训练模型
print("开始训练...")
with Timer() as train_time:
    model.fit()

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

# 构造只包含该用户的DataFrame
test_my_user = pd.DataFrame({
    'userID': [MY_USER_ID],
    'itemID': [0]  # 占位，模型不会使用这个值
})

# 生成推荐
topk_scores = model.recommend_k_items(test_my_user, top_k=TOP_K, remove_seen=True)
print("推荐完成！")

# %% [code]
# 展示推荐结果（含电影名称）
print(f"=== 为用户 {MY_USER_ID} 的 Top-{TOP_K} 推荐 ===")
print()

# 合并电影名称
user_recs = topk_scores[topk_scores['userID'] == MY_USER_ID].copy()
user_recs = user_recs.merge(movies_df[['itemID', 'title', 'genres']], on='itemID', how='left')

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

SAMPLE_USERS = 1000  # 采样用户数，可调整

# 随机选取测试集中的部分用户
test_user_sample = test['userID'].drop_duplicates().sample(n=SAMPLE_USERS, random_state=SEED)
test_sample = test[test['userID'].isin(test_user_sample)].copy()

print(f"测试集总用户数: {test['userID'].nunique()}")
print(f"采样用户数: {SAMPLE_USERS}")
print(f"采样数据量: {len(test_sample)}")

# 为采样用户生成推荐
print("为采样用户生成推荐...")
all_topk_scores = model.recommend_k_items(test_sample, top_k=TOP_K, remove_seen=True)
print("推荐完成！")

# %% [code]
# 评估模型性能（基于采样数据）
print("评估模型性能（基于采样数据）...")
print()

eval_map = map(test_sample, all_topk_scores, k=TOP_K)
eval_ndcg = ndcg_at_k(test_sample, all_topk_scores, k=TOP_K)
eval_precision = precision_at_k(test_sample, all_topk_scores, k=TOP_K)
eval_recall = recall_at_k(test_sample, all_topk_scores, k=TOP_K)

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
print("LightGCN推荐系统 - 实验总结")
print("="*60)
print()
print("【数据集】")
print(f"  - 类型: MovieLens 20M")
print(f"  - 用户数: {data.n_users}")
print(f"  - 物品数: {data.n_items}")
print(f"  - 训练交互数: {len(train)}")
print(f"  - 测试交互数: {len(test)}")
print()
print("【个人用户】")
print(f"  - 用户ID: {MY_USER_ID}")
print(f"  - 训练集评分数: {len(my_train)}")
print(f"  - 测试集评分数: {len(my_test)}")
print()
print("【模型配置】")
print(f"  - 算法: LightGCN")
print(f"  - 嵌入维度: 64")
print(f"  - GCN层数: 3")
print(f"  - 损失函数: BPR")
print(f"  - 训练轮数: {EPOCHS}")
print(f"  - 学习率: 0.001")
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
