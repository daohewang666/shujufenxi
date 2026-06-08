import pandas as pd
# 模拟原版5字段数据，字段和原数据集一模一样
data = [
    [1001,88,25,"pv",1511544000],
    [1001,99,36,"cart",1511544120],
    [1002,77,25,"fav",1511544300],
    [1003,66,44,"buy",1511545000]
]
df = pd.DataFrame(data,columns=["user_id","item_id","category_id","behavior","timestamp"])
print(df.head())

