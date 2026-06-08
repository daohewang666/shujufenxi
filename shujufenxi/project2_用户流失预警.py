# ==============================
# 项目2：用户流失预警模型
# 技术：Python + 机器学习随机森林
# 作用：预测哪些用户会流失（企业真实应用）
# ==============================

import matplotlib
# 第一步：先导入plt
import matplotlib.pyplot as plt

# 第二步：紧跟在导入后面再加中文设置（就放这里！）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
matplotlib.use('Agg')
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# ----------------------
# 1. 构建用户数据（真实业务特征）
# ----------------------
data = {
    "活跃度": ["高","中","高","低","低","中","高","低","中","高"] * 1000,
    "近30天登录天数": [2,9,22,1,0,12,28,2,10,25] * 1000,
    "近30天消费金额": [0,260,950,40,0,380,1800,20,500,1500] * 1000,
    "近7天点击次数": [4,25,70,2,0,30,90,1,40,80] * 1000,
    "是否加购": [0,1,1,0,0,1,1,0,1,1] * 1000,
    "是否收藏": [0,0,1,0,0,1,1,0,1,1] * 1000,
    "是否下单": [0,1,1,0,0,1,1,0,1,1] * 1000,
    "是否流失": [1,0,0,1,1,0,0,1,0,0] * 1000  # 1=流失 0=未流失
}

df = pd.DataFrame(data)
print("✅ 数据加载完成，共", len(df), "条用户数据")

# ----------------------
# 2. 数据处理（文字转数字）
# ----------------------
le = LabelEncoder()
df["活跃度"] = le.fit_transform(df["活跃度"])

# ----------------------
# 3. 拆分特征和标签
# ----------------------
X = df.drop("是否流失", axis=1)
y = df["是否流失"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# ----------------------
# 4. 训练机器学习模型
# ----------------------
model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)
print("\n✅ 模型训练完成！")

# ----------------------
# 5. 模型评估
# ----------------------
y_pred = model.predict(X_test)
print("\n==== 模型评估报告 ====")
print(classification_report(y_test, y_pred))

# ----------------------
# 6. 特征重要性（最核心！）
# ----------------------
feature_imp = pd.Series(model.feature_importances_, index=X.columns)
feature_imp = feature_imp.sort_values(ascending=False)

print("\n==== 影响用户流失的关键因素 ====")
print(feature_imp)

# ----------------------
# 7. 画图保存
# ----------------------
plt.figure(figsize=(10,5))
feature_imp.plot(kind='bar', color='#ff5555')
plt.title("用户流失影响因素")
plt.tight_layout()
plt.savefig("./data/流失因素图.png", dpi=150)
plt.close()

print("\n🎉 项目2 全部完成！图片已保存！")