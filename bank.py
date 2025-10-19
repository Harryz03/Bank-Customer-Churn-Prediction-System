#####⚠️代码请在终端中输入以下指令运行，请勿直接在pycharm等编译器中运行！！！！！！！！！！
##### windows使用这条指令：streamlit run "C:\path\to\your\script\script_name.py"
##### MacOS及Linux使用这条指令：streamlit run /path/to/your/script/script_name.py
##### streamlit run /Users/zhangruiqing/PycharmProjects/pythonProject2/BANK_Final/bank.py
import openml
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from matplotlib import rcParams
import pickle

# --- 配置 Matplotlib ---
rcParams['font.sans-serif'] = ['Arial']  # 使用默认字体避免中文问题
rcParams['axes.unicode_minus'] = False

# --- Streamlit 页面配置 ---
st.set_page_config(
    page_title="银行客户流失率预测系统",
    layout="wide"
)

st.title("银行客户流失率预测系统")

# 定义特征列名
feature_columns = ["a1cx", "a1cy", "b2x", "b2y", "a2pop", "a3pop", "temp", "mxql", "rej"]

# 创建一个包含这些列名的空模板
feature_template = pd.DataFrame(columns=feature_columns)

# 将模板转换为CSV文件，供用户下载
csv = feature_template.to_csv(index=False)

# --- 提供列名模板供用户下载 ---
def download_feature_template():
    # 下载模板文件的 CSV 数据
    csv_data = feature_template.to_csv(index=False)

    # 使用 Streamlit 的 HTML 来排版
    st.markdown(
        """
        ### 下载模板
        <div style="display: flex; align-items: center;">
        </div>
        """,
        unsafe_allow_html=True
    )

    # 添加下载按钮
    st.download_button(
        label="下载",
        data=csv_data,
        file_name="feature_template.csv",
        mime="text/csv",
    )

    # 设置状态变量
    if "show_meaning" not in st.session_state:
        st.session_state["show_meaning"] = False

    # 列名解释功能
    def toggle_meaning():
        st.session_state["show_meaning"] = not st.session_state["show_meaning"]

    # 添加交互逻辑
    clicked = st.button("列名解释", on_click=toggle_meaning)

    # 显示列名解释
    if st.session_state["show_meaning"]:

        # 定义列名含义字典
        column_meaning = {
            "a1cx,alcy": "客户所在位置",
            "b2x,b2y": "当前队列状态",
            "a2pop,a3pop": "客户耐心水平",
            "temp,mxql": "柜员效率",
            "rej": "拒绝率"
        }

        # 渲染列名解释表
        for col, meaning in column_meaning.items():
            st.write(f"**{col}**: {meaning}")

# --- 1. 训练数据加载与模型保存 ---
@st.cache_data
def load_training_data():
    dataset = openml.datasets.get_dataset(572)
    data, _, _, _ = dataset.get_data()
    return data

def train_and_save_models():
    data = load_training_data()
    X = data[feature_columns[:-1]]
    y = data["rej"]

    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 模型训练
    gbrt = GradientBoostingRegressor(random_state=42).fit(X_train, y_train)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42).fit(X_train, y_train)
    lgb_model = lgb.LGBMRegressor(random_state=42).fit(X_train, y_train)

    # 保存模型
    with open("gbrt_model.pkl", "wb") as f:
        pickle.dump(gbrt, f)
    with open("xgb_model.pkl", "wb") as f:
        pickle.dump(xgb_model, f)
    with open("lgb_model.pkl", "wb") as f:
        pickle.dump(lgb_model, f)

    st.success("模型已训练完成并保存！")


# --- 2. 文件上传功能 ---
def upload_file():
    st.write("### 上传文件")
    uploaded_file = st.file_uploader("上传CSV文件 (确保列名与模板一致)", type=["csv"])
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)

        # 校验列名：处理多余列与缺失列
        missing_cols = set(feature_columns) - set(user_data.columns)
        extra_cols = set(user_data.columns) - set(feature_columns)

        # 处理多余列：忽略
        if extra_cols:
            st.warning(f"检测到以下多余列，将被忽略：{extra_cols}")
            user_data = user_data.drop(columns=extra_cols)

        # 处理缺失列：填充默认值
        if missing_cols:
            st.warning(f"检测到以下缺失列，将自动填充默认值（0）：{missing_cols}")
            for col in missing_cols:
                user_data[col] = 0

        # 进一步检查并填充所有数据中的 NaN 值
        user_data = user_data.fillna(0)

        # 校验列名是否一致
        if set(user_data.columns) == set(feature_columns):
            st.success("上传文件的列名与模板一致，已成功加载数据。")
            return user_data  # 返回上传的数据
        else:
            st.error("列名处理失败，请检查文件内容！")
            return None
    return None


# --- 3. 加载模型和可视化功能 ---
def run_model_analysis(data):

    # 加载模型
    with open("gbrt_model.pkl", "rb") as f:
        gbrt = pickle.load(f)
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("lgb_model.pkl", "rb") as f:
        lgb_model = pickle.load(f)

    # 加载标准化器
    scaler = StandardScaler()
    X = data[feature_columns[:-1]]
    y = data["rej"]

    # 对上传的数据进行标准化
    X_test = scaler.fit_transform(X)
    y_test = y

    # 计算流失的客户数和流失率
    total_customers = len(data)
    churned_customers = data['rej'].sum()  # 'rej'列表示是否流失
    churn_rate = churned_customers / total_customers * 100

    # 在页面上显示流失统计信息
    st.write(f"### 客户流失统计")
    st.write(f"总客户数: {total_customers}")
    st.write(f"流失客户数: {churned_customers}")
    st.write(f"流失率: {churn_rate:.2f}%")

    # 模型评价函数
    def evaluate_model(model, X_test, y_test):
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        ev = explained_variance_score(y_test, y_pred)
        return y_pred, {"MAE": mae, "MSE": mse, "RMSE": rmse, "R²": r2, "Explained Variance": ev}

    # 模型评价
    metrics = {}
    for model_name, model in zip(["GBRT", "XGBoost", "LightGBM"], [gbrt, xgb_model, lgb_model]):
        y_pred, metric = evaluate_model(model, X_test, y_test)
        metrics[model_name] = metric

    st.write("### 模型评价结果")
    metrics_df = pd.DataFrame(metrics).T
    st.dataframe(metrics_df)

    # 数据可视化
    def plot_error_distribution(y_test, y_pred, ax):
        sns.histplot(y_test - y_pred, kde=True, ax=ax, color="blue", alpha=0.7)
        ax.set_title("Error Distribution")
        ax.set_xlabel("Error")
        ax.set_ylabel("Frequency")

    def plot_predictions(y_test, y_pred, ax):
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, ax=ax)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', color="red")
        ax.set_title("Actual vs Predicted")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")

    def plot_target_distribution(y_pred, ax):
        sns.histplot(y_pred, kde=True, ax=ax, color="green", alpha=0.7)
        ax.set_title("Predicted Target Distribution")
        ax.set_xlabel("Predicted Value")
        ax.set_ylabel("Frequency")

    st.write("### 数据可视化")
    for model_name, model in zip(["GBRT", "XGBoost", "LightGBM"], [gbrt, xgb_model, lgb_model]):
        y_pred, _ = evaluate_model(model, X_test, y_test)
        st.write(f"**{model_name}**")
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(4, 4))
            plot_error_distribution(y_test, y_pred, ax)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            plot_predictions(y_test, y_pred, ax)
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(figsize=(4, 4))
            plot_target_distribution(y_pred, ax)
            st.pyplot(fig)


# --- 主逻辑 ---
download_feature_template()  # 显示模板下载
uploaded_data = upload_file()  # 上传文件

if uploaded_data is not None:
    run_model_analysis(uploaded_data)  # 如果文件上传成功，运行模型分析