# %% [markdown]
# # 推荐算法预测实践
# 基于 Microsoft Recommenders 库 (https://github.com/recommenders-team/recommenders)
# 本 Notebook 演示完整的推荐系统工作流程：
# 1. 数据准备（MovieLens 数据集）
# 2. 数据拆分（训练集 / 测试集）
# 3. SAR 模型训练
# 4. 生成 Top-K 推荐
# 5. 模型评估
# 6. 模型保存与加载
# 环境要求：
#   pip install "recommenders"
#   或在 Kaggle/Colab：%pip install -q "recommenders"

# %% [code]
%pip install -q "recommenders"
%pip install -q pandas numpy scikit-learn matplotlib

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display

from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_random_split
from recommenders.models.sar import SAR
from recommenders.evaluation.python_evaluation import (
    map,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

print(f"System: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print("Libraries loaded successfully!")

# %% [markdown]
# ## 一、数据准备
# 使用 MovieLens 100K 数据集：
# - 943 个用户，1682 部电影
# - 100,000 条评分记录（1-5 分）

# %% [code]
DATA_SIZE = "100k"

df = movielens.load_pandas_df(size=DATA_SIZE)
df["rating"] = df["rating"].astype(np.float32)

print(f"Dataset: MovieLens {DATA_SIZE}")
print(f"Total records: {len(df)}")
print(f"Users: {df['userID'].nunique()}, Items: {df['itemID'].nunique()}")
print(f"Sparsity: {1 - len(df) / (df['userID'].nunique() * df['itemID'].nunique()):.2%}")
print("\nSample data:")
display(df.head(10))
print(f"\nRating distribution:\n{df['rating'].value_counts().sort_index()}")

# %% [markdown]
# ## 二、数据拆分（训练集 / 测试集）
# 使用随机拆分（80% 训练，20% 测试）。

# %% [code]
TOP_K = 10

train_df, test_df = python_random_split(
    df,
    ratio=0.75,
    seed=42,
)

print(f"Train set: {len(train_df)} records ({len(train_df)/len(df):.1%})")
print(f"Test set:  {len(test_df)} records ({len(test_df)/len(df):.1%})")
print(f"Train unique users: {train_df['userID'].nunique()}")
print(f"Test unique users:  {test_df['userID'].nunique()}")

# %% [markdown]
# ## 三、SAR 模型训练
# SAR（Simple Algorithm for Recommendation）原理：
# 1. 计算物品-物品相似度矩阵（Jaccard / Lift / Count）
# 2. 推荐得分 = Σ 历史物品相似度 × 评分
# 3. 支持时间衰减：score × exp(-decay × Δt)
#
# 优势：无需迭代训练，速度快，可解释性强。

# %% [code]
sar = SAR(
    col_user="userID",
    col_item="itemID",
    col_rating="rating",
    similarity_type="jaccard",
    time_decay_coefficient=0.1,
    timedecay_formula=True,
)

print("Training SAR model...")
sar.fit(train_df)

print("Model trained successfully!")
print(f"Item similarity matrix shape: {sar.item_similarity.shape}")
print(f"Number of users: {sar.n_users}")
print(f"Number of items: {sar.n_items}")

# %% [markdown]
# ## 四、生成推荐（预测）
# 对每个测试用户，生成 Top-K 推荐物品列表。
# `remove_seen=True`：过滤用户已交互过的物品。

# %% [code]
print(f"Generating Top-{TOP_K} recommendations...")

top_k_recs = sar.recommend_k_items(
    test_df,
    top_k=TOP_K,
    remove_seen=True,
)

print(f"Recommendations generated for {top_k_recs['userID'].nunique()} users")
print("\nSample recommendations (first 20 rows):")
display(top_k_recs.head(20))

# Add rank column for each user's recommendations
top_k_recs['rank'] = top_k_recs.groupby('userID').cumcount() + 1

# %% [markdown]
# ## 五、推荐结果可视化

# %% [code]
EXAMPLE_USER = train_df["userID"].iloc[0]

print(f"=== User {EXAMPLE_USER} ===\n")

user_history = train_df[train_df["userID"] == EXAMPLE_USER].sort_values("rating", ascending=False)
print("Historical interactions (top rated):")
display(user_history.head(10))

user_recs = top_k_recs[top_k_recs["userID"] == EXAMPLE_USER].sort_values("rank")
print(f"\nTop-{TOP_K} recommendations:")
display(user_recs)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

user_all = df[df["userID"] == EXAMPLE_USER]
axes[0].hist(user_all["rating"], bins=5, edgecolor="black", alpha=0.7)
axes[0].set_title(f"User {EXAMPLE_USER} - Rating Distribution")
axes[0].set_xlabel("Rating")
axes[0].set_ylabel("Count")

if not user_recs.empty:
    axes[1].bar(user_recs["rank"], user_recs["prediction"], alpha=0.7)
    axes[1].set_title(f"User {EXAMPLE_USER} - Top {TOP_K} Scores")
    axes[1].set_xlabel("Rank")
    axes[1].set_ylabel("Prediction Score")

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 六、模型评估
# 评估指标说明：
# | 指标 | 类型 | 说明 |
# |------|------|------|
# | Precision@K | 排序 | 推荐的前 K 个中有多少在测试集里 |
# | Recall@K    | 排序 | 测试集里的物品有多少被推荐了 |
# | nDCG@K      | 排序 | 考虑排序位置的加权 Recall |
# | MAP@K       | 排序 | 平均精度（综合 P@1..P@K）|

# %% [code]
print("Evaluating model performance...\n")

eval_map = map(test_df, top_k_recs, col_user="userID", col_item="itemID", col_rating="rating", k=TOP_K)
eval_ndcg = ndcg_at_k(test_df, top_k_recs, col_user="userID", col_item="itemID", col_rating="rating", k=TOP_K)
eval_precision = precision_at_k(test_df, top_k_recs, col_user="userID", col_item="itemID", col_rating="rating", k=TOP_K)
eval_recall = recall_at_k(test_df, top_k_recs, col_user="userID", col_item="itemID", col_rating="rating", k=TOP_K)

print(f"=== Evaluation Results (Top-{TOP_K}) ===")
print(f"  MAP@K:      {eval_map:.4f}")
print(f"  nDCG@K:     {eval_ndcg:.4f}")
print(f"  Precision@K: {eval_precision:.4f}")
print(f"  Recall@K:    {eval_recall:.4f}")

# %% [markdown]
# ## 七、不同算法对比
# Recommenders 库支持 30+ 种算法：
#
# | 算法 | 类型 | 速度 | 精度 | 适用场景 |
# |------|------|------|------|----------|
# | SAR       | 协同过滤 | ⚡⚡⚡ | ⭐⭐ | 快速原型、冷启动 |
# | ALS       | 矩阵分解 | ⚡⚡   | ⭐⭐⭐ | 大规模（需 Spark）|
# | NCF       | 深度学习 | ⚡     | ⭐⭐⭐⭐ | 隐式反馈 |
# | LightGCN  | 图神经网络 | ⚡   | ⭐⭐⭐⭐ | 序列推荐 |
# | Wide&Deep | 深度学习 | ⚡⚡   | ⭐⭐⭐⭐ | 特征丰富场景 |

# %% [code]
print("=" * 60)
print("NCF (Neural Collaborative Filtering) - Code Structure")
print("=" * 60)

NCF_EXAMPLE = """
from recommenders.models.ncf.ncf_singlenode import NCF

ncf_model = NCF(
    n_users=df['userID'].nunique(),
    n_items=df['itemID'].nunique(),
    n_factors=64,
    model_type='NeuMF',
    n_neg=4,
    lr=0.001,
    epochs=10,
)

ncf_model.fit(train_df, test_df)

ncf_recs = ncf_model.recommend_k_items(test_df, top_k=10, remove_seen=True)
"""

print(NCF_EXAMPLE)
print("Note: NCF requires GPU for best performance.")

# %% [markdown]
# ## 八、模型保存与加载

# %% [code]
import pickle
import os

MODEL_DIR = "./saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

model_path = os.path.join(MODEL_DIR, "sar_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump({
        "item_similarity": sar.item_similarity,
        "user2index": sar.user2index,
        "item2index": sar.item2index,
        "n_users": sar.n_users,
        "n_items": sar.n_items,
    }, f)

print(f"Model saved to: {model_path}")

with open(model_path, "rb") as f:
    loaded = pickle.load(f)

print(f"Model loaded! Similarity matrix shape: {loaded['item_similarity'].shape}")

# %% [markdown]
# ## 九、实战：为指定用户生成推荐

# %% [code]
def get_recommendations_for_user(user_id, model, train_data, top_k=10):
    """为指定用户生成推荐列表。"""
    user_history = set(train_data[train_data["userID"] == user_id]["itemID"].unique())
    all_items = set(train_data["itemID"].unique())
    candidate_items = list(all_items - user_history)

    if len(candidate_items) == 0:
        return pd.DataFrame(columns=["userID", "itemID", "prediction", "rank"])

    test_input = pd.DataFrame({
        "userID": [user_id] * len(candidate_items),
        "itemID": candidate_items,
        "rating": [0] * len(candidate_items),
    })

    recs = model.recommend_k_items(test_input, top_k=top_k, remove_seen=True)
    user_recs = recs[recs["userID"] == user_id].copy()
    user_recs['rank'] = range(1, len(user_recs) + 1)
    return user_recs.sort_values("rank")


target_user = train_df["userID"].iloc[0]
user_recs = get_recommendations_for_user(target_user, sar, train_df, top_k=5)

print(f"=== Recommendations for User {target_user} ===")
if not user_recs.empty:
    display(user_recs)
else:
    print("No recommendations generated.")

# %% [markdown]
# ## 十、总结与扩展方向
# 已完成：
# - ✅ 数据加载与拆分
# - ✅ SAR 模型训练
# - ✅ Top-K 推荐生成
# - ✅ 模型评估（MAP / nDCG / Precision / Recall）
# - ✅ 模型保存与加载
# 扩展方向：
# 1. 加入物品特征（电影类型）→ Wide & Deep
# 2. 序列推荐 → SASRec / GRU4Rec
# 3. 冷启动处理 → 热门推荐 / 内容相似度
# 参考：https://github.com/recommenders-team/recommenders

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
