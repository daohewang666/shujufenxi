# 最顶部加两行，禁用弹窗绘图，只生成图片文件
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ==========第一步：生成模拟70w量级小样数据，自动保存到data文件夹==========
# 模拟原始5列：user_id,item_id,category_id,behavior,timestamp
behavior_list = ["pv","cart","fav","buy"]
import random,time
random.seed(1)
sample_rows = []
start_ts = int(datetime.strptime("2017-11-01","%Y-%m-%d").timestamp())
end_ts = int(datetime.strptime("2017-12-01","%Y-%m-%d").timestamp())
for _ in range(80000):
    uid = random.randint(1000,50000)
    iid = random.randint(100,99999)
    cid = random.randint(1,5000)
    act = random.choice(behavior_list)
    ts = random.randint(start_ts,end_ts)
    sample_rows.append([uid,iid,cid,act,ts])
df = pd.DataFrame(sample_rows,columns=["user_id","item_id","category_id","behavior","timestamp"])
# 保存csv到data目录
df.to_csv("./data/UserBehavior.csv",index=False)
print("数据集已自动存入data文件夹！")

# ==========第二步：读取数据+时间戳转日期（简历内容：数据清洗）==========
df = pd.read_csv("./data/UserBehavior.csv")
# 时间戳转日期、小时
df["date"] = pd.to_datetime(df["timestamp"],unit="s").dt.date
df["hour"] = pd.to_datetime(df["timestamp"],unit="s").dt.hour

# ==========第三步：计算DAU日活（简历指标1）==========
dau_df = df.groupby("date")["user_id"].nunique().reset_index()
print("====每日DAU数据====")
print(dau_df.head())

# ==========第四步：转化漏斗：浏览→收藏→加购→下单（简历核心）==========
pv_cnt = len(df[df["behavior"]=="pv"])
cart_cnt = len(df[df["behavior"]=="cart"])
fav_cnt = len(df[df["behavior"]=="fav"])
buy_cnt = len(df[df["behavior"]=="buy"])
funnel_data = [pv_cnt,cart_cnt,fav_cnt,buy_cnt]
funnel_name = ["浏览","加购","收藏","下单"]
print("\n====转化漏斗各环节人数====")
for n,c in zip(funnel_name,funnel_data):
    print(f"{n}:{c}")

# ==========第五步：画漏斗图（生成图片，放进作品集）==========
plt.rcParams["font.sans-serif"]=["SimHei"]
plt.figure(figsize=(8,5))
plt.bar(funnel_name,funnel_data,color=["#ff7777","#ffbb66","#77bbff","#77dd77"])
plt.title("用户全链路转化漏斗图")
plt.savefig("./data/漏斗图1.png",dpi=150)
plt.close() # 关闭画布，释放内存，删掉plt.show()
print("漏斗图已保存至data文件夹！")

# ==========第六步：计算次日留存（简历留存指标）==========
# 筛选首日活跃用户
first_day = df["date"].min()
user_first = set(df[df["date"]==first_day]["user_id"])
# 次日活跃用户
next_day = df[df["date"]==pd.Timestamp(first_day)+pd.Timedelta(days=1)]
user_next = set(next_day["user_id"])
retain_user = user_first & user_next
retain_rate = len(retain_user)/len(user_first)
print(f"\n次日留存率：{round(retain_rate*100,2)}%")