# ==============================
# 项目3：RFM 用户价值分层模型（已修复报错版）
# ==============================
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
import random
from datetime import datetime, timedelta

# 解决中文乱码
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------------
# 1. 生成真实业务数据
# ----------------------
random.seed(1)
now = datetime.now()
data = []

for i in range(1, 8001):
    user_id = i
    days = random.randint(1, 60)
    R = (now - timedelta(days=days)).date()
    F = random.randint(1, 30)
    M = random.randint(10, 5000)
    data.append([user_id, R, F, M])

df = pd.DataFrame(data, columns=["user_id", "recent_date", "F", "M"])

# 【已修复】把日期转成标准时间格式，避免报错
df["recent_date"] = pd.to_datetime(df["recent_date"])
now = pd.to_datetime(now)

# 计算距离今天多少天没买（R）
df["R"] = (now - df["recent_date"]).dt.days

print("✅ 数据生成完成：", len(df), "条用户数据")

# ----------------------
# 2. RFM 打分（0-5分）
# ----------------------
df["R_score"] = pd.cut(df["R"], bins=[0, 7, 15, 30, 60, 9999], labels=[5,4,3,2,1])
df["F_score"] = pd.cut(df["F"], bins=[0, 5, 10, 15, 25, 9999], labels=[1,2,3,4,5])
df["M_score"] = pd.cut(df["M"], bins=[0, 200, 500, 1000, 3000, 99999], labels=[1,2,3,4,5])

df["R_score"] = df["R_score"].astype(int)
df["F_score"] = df["F_score"].astype(int)
df["M_score"] = df["M_score"].astype(int)

# ----------------------
# 3. 计算 RFM 均值，判断高低
# ----------------------
r_mean = df["R_score"].mean()
f_mean = df["F_score"].mean()
m_mean = df["M_score"].mean()

def rfm_type(row):
    r = 1 if row["R_score"] > r_mean else 0
    f = 1 if row["F_score"] > f_mean else 0
    m = 1 if row["M_score"] > m_mean else 0

    if r==1 and f==1 and m==1:
        return "高价值用户"
    elif r==1 and f==1 and m==0:
        return "潜力用户"
    elif r==1 and f==0 and m==1:
        return "高频大额用户"
    elif r==0 and f==1 and m==1:
        return "沉睡高价值用户"
    elif r==1 and f==0 and m==0:
        return "新用户"
    elif r==0 and f==1 and m==0:
        return "低频老用户"
    elif r==0 and f==0 and m==1:
        return "大额流失用户"
    else:
        return "流失用户"

df["用户类型"] = df.apply(rfm_type, axis=1)

# ----------------------
# 4. 统计各类用户数量
# ----------------------
user_type_count = df["用户类型"].value_counts()
print("\n==== RFM 用户分层结果 ====")
print(user_type_count)

# ----------------------
# 5. 画图保存
# ----------------------
plt.figure(figsize=(12,6))
user_type_count.plot(kind="bar", color="#ff6699")
plt.title("RFM 用户分群数量分布")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("./data/RFM用户分层.png", dpi=150)
plt.close()

print("\n🎉 项目3 全部完成！图片已保存！")