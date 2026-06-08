# ==============================
# 项目4：商品销量预测模型（线性回归）
# 机器学习回归任务 → 预测连续数值（销量）
# ==============================
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 解决中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------
# 1. 生成商品数据（价格、促销、广告、点击、销量）
# ----------------------
np.random.seed(1)
data = {
    "商品价格": np.random.randint(39, 599, 1000),       # 价格39~599
    "是否促销": np.random.randint(0, 2, 1000),         # 0=不促销 1=促销
    "广告投入": np.random.randint(50, 2000, 1000),     # 广告费50~2000
    "页面点击量": np.random.randint(100, 15000, 1000), # 点击量
    "好评数": np.random.randint(10, 2000, 1000),       # 好评数
}

# 销量 = 受点击、广告、好评正向影响，价格负向影响
df = pd.DataFrame(data)
df["销量"] = (
    0.3 * df["页面点击量"]
    + 0.2 * df["广告投入"]
    + 0.25 * df["好评数"]
    - 0.6 * df["商品价格"]
    + 200
    + np.random.randint(-50, 50, 1000)
)
df["销量"] = df["销量"].astype(int)
df.loc[df["销量"] < 0, "销量"] = 10

print("✅ 商品数据生成完成：", len(df), "条")

# ----------------------
# 2. 拆分特征 X 和 标签 y
# ----------------------
X = df.drop("销量", axis=1)
y = df["销量"]

# 训练集80%，测试集20%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# ----------------------
# 3. 训练线性回归模型
# ----------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------
# 4. 预测 & 评估
# ----------------------
y_pred = model.predict(X_test)

# 评估指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n==== 模型评估 ====")
print(f"均方误差 (MSE)：{round(mse, 2)}")
print(f"决定系数 (R²)：{round(r2, 2)}")

# 特征重要性（回归系数）
coef = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)
print("\n==== 影响销量的关键因素（系数越大影响越大）====")
print(coef)

# ----------------------
# 5. 画图：真实销量 vs 预测销量
# ----------------------
plt.figure(figsize=(10,5))
plt.scatter(y_test, y_pred, alpha=0.5, color="#4488ff")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.xlabel("真实销量")
plt.ylabel("预测销量")
plt.title("真实销量 vs 预测销量")
plt.tight_layout()
plt.savefig("./data/销量预测对比图.png", dpi=150)
plt.close()

# ----------------------
# 6. 特征重要性图
# ----------------------
plt.figure(figsize=(10,5))
coef.plot(kind="bar", color="#ff6666")
plt.title("影响销量的因素重要性")
plt.tight_layout()
plt.savefig("./data/销量影响因素.png", dpi=150)
plt.close()

print("\n🎉 项目4 全部完成！两张图片已保存！")