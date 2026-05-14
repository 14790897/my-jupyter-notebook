# %% [markdown]
# # 推荐算法预测实践
# 基于 **Microsoft Recommenders** 库 (https://github.com/recommenders-team/recommenders)
# 本 Notebook 演示完整的推荐系统工作流程：
# 1. 数据准备（MovieLens 数据集）
# 2. 模型训练（SAR - Simple Algorithm for Recommendation）
# 3. 生成推荐结果
# 4. 模型评估
# **环境要求**：
# - Python 3.8+
# - `pip install "recommenders[all]"`
# - 或 CPU 轻量安装：`pip install recommenders`
# **算法说明（SAR）**：
# - 基于物品相似度（Item-Item Collaborative Filtering）
# - 无需训练，适合快速原型
# - 支持隐式反馈（点击、购买、评分）

# %% [code]
# ========== 安装依赖 ==========
%pip install -q "recommenders"
%pip install -q pandas numpy scikit-learn

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Recommenders 库
from recommenders.datasets import movielens
from recommenders.models.sar import SAR
from recommenders.utils.evaluate import (
    eval_top_k,
    rmse,
    mae,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
)

print("Libraries loaded successfully!")
print(f"Pandas version: {pd.__version__}")

# %% [markdown]
# ## 一、数据准备
# 使用 **MovieLens 100K** 数据集：
# - 943 个用户，1682 部电影
# - 100,000 条评分记录（1-5 分）
# - 稀疏度约 94%
# 数据格式要求：
# ```
# | userID | itemID | rating | timestamp |
# ```

# %% [code]
# ========== 加载 MovieLens 数据 ==========
# 数据会自动下载到 ~/.recommenders/datasets/
DATA_SIZE = "100k"  # 可选: "100k", "1m", "10m", "20m"

df = movielens.load_pandas_df(
    size=DATA_SIZE,
    header=["userID", "itemID", "rating", "timestamp"],
    local_cache_path="./ml-100k/",  # 本地缓存路径
)

print(f"Dataset: MovieLens {DATA_SIZE}")
print(f"Total records: {len(df)}")
print(f"Users: {df['userID'].nunique()}, Items: {df['itemID'].nunique()}")
print("\nSample data:")
display(df.head(10))

# 数据基本统计
print("\nData statistics:")
print(f"  Sparsity: {1 - len(df) / (df['userID'].nunique() * df['itemID'].nunique()):.2%}")
print(f"  Rating distribution:\n{df['rating'].value_counts().sort_index()}")

# %% [markdown]
# ## 二、数据拆分（训练集 / 测试集）
# 按时间顺序拆分（更贴近真实推荐场景）：
# - 前 80% 时间的数据 → 训练集
# - 后 20% 时间的数据 → 测试集
# 也可使用随机拆分，适合基线对比。

# %% [code]
# ========== 时间序列拆分 ==========
df_sorted = df.sort_values(["userID", "timestamp"]).reset_index(drop=True)

# 每个用户的训练/测试拆分
train_list = []
test_list = []

for user_id, user_data in df_sorted.groupby("userID"):
    n_train = int(len(user_data) * 0.8)
    train_list.append(user_data.iloc[:n_train])
    test_list.append(user_data.iloc[n_train:])

train_df = pd.concat(train_list, ignore_index=True)
test_df = pd.concat(test_list, ignore_index=True)

print(f"Train set: {len(train_df)} records ({len(train_df)/len(df):.1%})")
print(f"Test set:  {len(test_df)} records ({len(test_df)/len(df):.1%})")
print(f"\nTrain unique users: {train_df['userID'].nunique()}")
print(f"Test unique users:  {test_df['userID'].nunique()}")

# %% [markdown]
# ## 三、SAR 模型训练
# **SAR（Simple Algorithm for Recommendation）** 原理：
# 1. **物品-物品相似度矩阵**：
#    ```
#    S[i][j] = 余弦相似度(Affinity_i, Affinity_j)
#    ```
#    其中 Affinity 是用户对物品的交互向量
#
# 2. **推荐得分**：
#    ```
#    score(u, i) = Σ_{j ∈ 用户u的历史物品} S[i][j] × rating(u, j)
#    ```
# 3. **时间衰减**（可选）：
#    ```
#    weight = exp(-decay_rate × Δt)
#    ```
# **优势**：
# - 无需迭代训练，速度极快
# - 可解释性强（推荐理由 = 相似物品）
# - 支持新用户（冷启动 via 热门物品）

# %% [code]
# ========== 初始化 SAR 模型 ==========
sar = SAR(
    col_user="userID",
    col_item="itemID",
    col_rating="rating",
    col_timestamp="timestamp",
    similarity_type="jaccard",   # 相似度类型: "jaccard" | "lift" | "count"
    time_decay_coefficient=0.1,  # 时间衰减系数 (0=无衰减)
    timedecay_formula=True,        # 是否使用时间衰减
)

# 训练（实际上是在计算物品相似度矩阵）
print("Training SAR model...")
sar.fit(train_df)

print("Model trained successfully!")
print(f"Item similarity matrix shape: {sar.item_similarity.shape}")
print(f"Number of items: {sar.n_items}")
print(f"Number of users: {sar.n_users}")

# %% [markdown]
# ## 四、生成推荐（预测）
# 对测试集中的用户，预测他们最可能喜欢的 Top-K 物品。
# **预测流程**：
# 1. 对每个用户，找到其历史交互物品
# 2. 计算候选物品得分：`score = Σ similarity × rating`
# 3. 排除已交互物品
# 4. 取 Top-K 得分物品作为推荐结果
# %% [code]
# ========== 生成推荐 ==========
TOP_K = 10  # 每个用户推荐 K 个物品

print(f"Generating Top-{TOP_K} recommendations for each user in test set...")

# 对测试集中的所有用户生成推荐
recommendations = sar.recommend_k_users(
    test_df,
    top_k=TOP_K,
    remove_seen=True,  # 移除用户已交互过的物品
)

print(f"Recommendations generated for {recommendations['userID'].nunique()} users")
print("\nSample recommendations:")
display(recommendations.head(20))

# 推荐结果格式：
# | userID | itemID | prediction | rank |
# prediction = 推荐得分（越高越好）

# %% [markdown]
# ## 五、推荐结果可视化
# 查看具体用户的推荐结果，并对比其历史交互。

# %% [code]
# ========== 查看某用户的推荐详情 ==========
EXAMPLE_USER = train_df["userID"].iloc[0]  # 选第一个用户作为示例

print(f"=== User {EXAMPLE_USER} ===\n")

# 该用户的历史交互（训练集）
user_history = train_df[train_df["userID"] == EXAMPLE_USER].sort_values("rating", ascending=False)
print("Historical interactions (top rated):")
display(user_history.head(10))

# 给该用户的推荐
user_recs = recommendations[recommendations["userID"] == EXAMPLE_USER].sort_values("rank")
print(f"\nTop-{TOP_K} recommendations:")
display(user_recs)

# 可视化：该用户的评分分布
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# 评分分布
user_all = df[df["userID"] == EXAMPLE_USER]
axes[0].hist(user_all["rating"], bins=5, edgecolor="black", alpha=0.7)
axes[0].set_title(f"User {EXAMPLE_USER} - Rating Distribution")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Count")

# 推荐得分分布
axes[1].bar(user_recs["rank"], user_recs["prediction"], alpha=0.7)
axes[1].set_title(f"User {EXAMPLE_USER} - Top {TOP_K} Recommendation Scores")
axes[1].set_xlabel("Rank")
axes[1].set_ylabel("Prediction Score")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 六、模型评估
# 使用以下指标评估推荐质量：
# | 指标 | 说明 |
# |------|------|
# | **Precision@K** | 推荐的前 K 个物品中，有多少在测试集里（准确率） |
# | **Recall@K** | 测试集里的物品，有多少出现在推荐的前 K 个里（召回率） |
# | **nDCG@K** | 考虑排序位置的加权平均（越高越好） |
# | **MAP@K** | 平均精度均值（综合考虑排序质量） |
# | **RMSE** | 评分预测误差（仅当有评分标注时） |

# %% [code]
# ========== 评估推荐质量 ==========
print("Evaluating model performance...")

# 将推荐结果和测试集合并，计算指标
eval_results = eval_top_k(
    recommendations,
    test_df,
    col_user="userID",
    col_item="itemID",
    k=TOP_K,
)

print(f"\n=== Evaluation Results (Top-{TOP_K}) ===")
for metric, value in eval_results.items():
    print(f"  {metric}: {value:.4f}")

# 各指标的详细解释
print("\nMetric explanations:")
print("  Precision@K  = 推荐中有多少是相关的（相关性 = 出现在测试集）")
print("  Recall@K     = 相关物品中有多少被推荐了")
print("  nDCG@K       = 考虑排序位置的加权 Recall（排前面权重更高）")
print("  MAP@K        = 平均精度，综合 P@1, P@2, ..., P@K")

# %% [markdown]
# ## 七、不同算法的对比（扩展）
# Recommenders 库支持 30+ 种算法，以下是常用算法的对比：
# | 算法 | 类型 | 训练速度 | 预测质量 | 适用场景 |
# |------|------|----------|----------|----------|
# | **SAR** | 协同过滤 | ⚡⚡⚡ | ⭐⭐ | 快速原型、冷启动 |
# | **ALS** | 矩阵分解 | ⚡⚡ | ⭐⭐⭐ | 大规模稀疏数据（需 Spark） |
# | **NCF** | 深度学习 | ⚡ | ⭐⭐⭐⭐ | 隐式反馈、复杂模式 |
# | **LightGCN** | 图神经网络 | ⚡ | ⭐⭐⭐⭐ | 序列推荐、社交推荐 |
# | **Wide & Deep** | 深度学习 | ⚡⚡ | ⭐⭐⭐⭐ | 特征丰富场景 |
# **下一步**：
# - 尝试 NCF 或 LightGCN 提升精度
# - 加入物品特征（类型、标签）→ 用 Wide & Deep
# - 处理序列数据（用户点击序列）→ 用 GRU4Rec / SASRec

# %% [code]
# ========== 算法对比示例（NCF vs SAR）==========
# 以下是 NCF（Neural Collaborative Filtering）的简化示例
# 完整代码见: recommenders/notebooks/00_quick_start/ncf_movielens.ipynb

print("=" * 60)
print("NCF (Neural Collaborative Filtering) - Quick Start")
print("=" * 60)

# NCF 需要 GPU 或较大内存，这里只展示代码结构：

NCF_CODE_EXAMPLE = """
from recommenders.models.ncf.ncf_singlenode import NCF

# 初始化 NCF 模型
ncf_model = NCF(
    n_users=df['userID'].nunique(),
    n_items=df['itemID'].nunique(),
    n_factors=64,          # 嵌入维度
    model_type='NeuMF',    # 'GMF' | 'MLP' | 'NeuMF'
    n_neg=4,               # 负采样数量
    lr=0.001,              # 学习率
    epochs=10,
)

# 训练
ncf_model.fit(train_df, test_df)

# 推荐
ncf_recs = ncf_model.recommend_k_items(test_df, top_k=10)
"""

print("NCF code structure (requires GPU for best performance):")
print(NCF_CODE_EXAMPLE)

# %% [markdown]
# ## 八、保存与加载模型
# 将训练好的 SAR 模型保存，方便后续直接加载使用。

# %% [code]
# ========== 保存模型 ==========
import pickle
import os

MODEL_DIR = "./saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# SAR 模型保存（物品相似度矩阵 + 用户/物品映射）
model_path = os.path.join(MODEL_DIR, "sar_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({
        "item_similarity": sar.item_similarity,
        "user_mapping": sar.user2index,
        "item_mapping": sar.item2index,
        "n_users": sar.n_users,
        "n_items": sar.n_items,
    }, f)

print(f"Model saved to: {model_path}")

# 加载模型
with open(model_path, "rb") as f:
    loaded = pickle.load(f)

print(f"Model loaded! Item similarity matrix shape: {loaded['item_similarity'].shape}")
print(f"Number of users in loaded model: {loaded['n_users']}")

# %% [markdown]
# ## 九、实战：为指定用户生成推荐
# 输入用户 ID，输出个性化推荐列表（可用于 API 接口）。

# %% [code]
# ========== 实战：推荐函数 ==========
def get_recommendations_for_user(user_id, model, train_data, top_k=10):
    """为指定用户生成推荐列表。

    Args:
        user_id: 用户 ID
        model: 训练好的 SAR 模型
        train_data: 训练数据（用于排除已交互物品）
        top_k: 推荐数量

    Returns:
        DataFrame: 推荐结果（itemID, prediction, rank）
    """
    # 获取该用户已交互的物品
    user_items = set(train_data[train_data["userID"] == user_id]["itemID"].unique())

    # 生成所有物品的推荐得分
    all_items = set(train_data["itemID"].unique())
    candidate_items = list(all_items - user_items)

    # 使用 SAR 的 predict 方法
    recs = model.recommend_k_items(
        pd.DataFrame({"userID": [user_id], "itemID": candidate_items}).head(len(candidate_items)),
        top_k=top_k,
        remove_seen=True,
    )

    return recs[recs["userID"] == user_id].sort_values("rank")


# 示例：为用户 1 生成推荐
target_user = train_df["userID"].iloc[0]
user_recs = get_recommendations_for_user(target_user, sar, train_df, top_k=5)

print(f"=== Personalized Recommendations for User {target_user} ===")
print(f"(Excluded {len(user_items)} already-interacted items)\n")
display(user_recs)

# %% [markdown]
# ## 十、总结与扩展方向
# **已完成**：
# - ✅ 数据加载与拆分
# - ✅ SAR 模型训练
# - ✅ Top-K 推荐生成
# - ✅ 模型评估（Precision/Recall/nDCG）
# - ✅ 模型保存与加载
# **扩展方向**：
# 1. **特征工程**：加入电影类型、用户画像（年龄、性别）
# 2. **序列推荐**：使用 GRU4Rec / SASRec 处理用户行为序列
# 3. **冷启动处理**：新用户 → 热门推荐；新物品 → 内容相似度
# 4. **A/B 测试**：在线评估推荐效果（点击率、转化率）
# 5. **部署**：用 Flask/FastAPI 封装推荐 API
# **参考资源**：
# - 官方文档: https://github.com/recommenders-team/recommenders
# - 算法 Notebook: `recommenders/notebooks/00_quick_start/`
# - 评估指标: `recommenders/utils/evaluate.py`

# %% [code]
print("=" * 60)
print("Recommendation Algorithm Practice - Complete!")
print("=" * 60)
print("\nNext steps:")
print("  1. Try NCF / LightGCN for better accuracy")
print("  2. Add item features (genres) for content-based filtering")
print("  3. Deploy the model as a REST API (FastAPI)")
print("  4. Evaluate online metrics (CTR, conversion rate)")
print("\nHappy recommending! 🎬")
