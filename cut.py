import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff

# 读取 ARFF 文件
input_file = "phpSRIcMD.arff"  # 确保文件路径正确
data, meta = arff.loadarff(input_file)

# 转换为 Pandas DataFrame
df = pd.DataFrame(data)

# 检查并解码字节型数据
for col in df.columns:
    if df[col].dtype == 'object':  # 如果是 object 类型，可能是字节型
        try:
            df[col] = df[col].str.decode('utf-8')  # 解码为字符串
        except AttributeError:
            pass  # 如果列不是字节型，跳过

# 分割数据集
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

# 保存测试集为 CSV 文件
test_set.to_csv("test_data_full.csv", index=False)

print("测试集已保存为 'test_data_full.csv'，第一行为特征名称。")